# H-Net (Diffusion / Non-Autoregressive) training/inference on Tiny Shakespeare
import argparse
import os
import time

import torch
from torch.nn import functional as F

from hnet_impl import HNetLM
import hnet_core as core

# -----------------
# hyperparameters
# -----------------
batch_size = 16
block_size = 256
max_iters = 4000
learning_rate = 3e-4
eval_interval = 200
eval_iters = 50
# --------------


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x]


@torch.no_grad()
def model_logits(model, x, vocab_size: int):
    iids = core.to_njt(x)
    logits, _extra = model(iids)
    return core.flat_logits_to_btt(logits, x.size(0), x.size(1), vocab_size)


@torch.no_grad()
def generate(
    model,
    max_new_tokens,
    prompt_len=16,
    temp=0.8,
    confidence_threshold=0.95,
    top_k=2,
):
    device = next(model.parameters()).device
    vocab = core.vocab_diff
    mask_token_id = vocab.mask_token_id

    all_tokens = core.train_data_diff[:prompt_len].tolist()
    total_steps = 0

    # Generate one block at a time
    while len(all_tokens) - prompt_len < max_new_tokens:
        block_len = min(240, prompt_len + max_new_tokens - len(all_tokens))

        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(all_tokens[-prompt_len:], device=device)

        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        while masked.any():
            total_steps += 1

            logits = model_logits(model, x, vocab.vocab_size)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(
                1, block_size
            )
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

        all_tokens.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    tokens_generated = len(all_tokens) - prompt_len
    print(f"Total steps: {total_steps} for {tokens_generated} tokens")
    print(f"Avg decoded per step: {tokens_generated / total_steps:.2f}")
    return vocab.decode(all_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H-Net diffusion LM")
    parser.add_argument("--train", action="store_true", help="Train from scratch")
    parser.add_argument(
        "--arch",
        type=str,
        default="T2,T2",
        help="Stage arch list, e.g. 'T2,T2' (avoid Mamba for diffusion)",
    )
    parser.add_argument(
        "--d-model",
        type=str,
        default="256,512",
        help="Stage model dims, e.g. '256,512'",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile H-Net blocks (CUDA only)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/hnet_diffusion.pt",
        help="Path to save/load model weights",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=2000,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=16,
        help="Prompt length",
    )
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temp")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence threshold for parallel decoding",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Top-k sampling")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("H-Net requires CUDA (flash-attn/mamba-ssm kernels).")

    arch = [a.strip() for a in args.arch.split(",") if a.strip()]
    d_model = parse_int_list(args.d_model)
    if len(arch) != len(d_model):
        raise ValueError("--arch and --d-model must have the same number of stages")

    # Build config (non-causal attention)
    c = core.build_hnet_config(
        vocab_size=core.vocab_diff.vocab_size,
        d_model=d_model,
        arch=arch,
        is_causal=False,
    )

    os.makedirs(os.path.dirname(args.weights), exist_ok=True)

    model = HNetLM(c).to(device)
    if args.compile:
        model.backbone.block_compile(ac=False)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    if os.path.exists(args.weights) and not args.train:
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("Training from scratch")

        optimizer = torch.optim.AdamW(
            [
                dict(params=ls, lr=learning_rate * lr_mod)
                for ls, lr_mod in zip(model.split_params_by_hierachy(), c.lambda_s())
            ],
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        start = time.time()
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = core.estimate_loss_diffusion(
                    model,
                    eval_iters=eval_iters,
                    batch_size=batch_size,
                    block_size=block_size,
                    vocab_size=core.vocab_diff.vocab_size,
                )
                print(
                    f"step {iter}: train loss {losses['train']:.4f},"
                    f"val loss {losses['val']:.4f}, time {time.time() - start:.2f} seconds"
                )
                sample = generate(
                    model,
                    max_new_tokens=240,
                    prompt_len=args.prompt_len,
                    temp=args.temp,
                    confidence_threshold=args.confidence,
                    top_k=args.top_k,
                )
                print(f"Sample:\n{sample}\n")

            xb, yb, mb = core.get_batch_diffusion(
                "train", batch_size=batch_size, block_size=block_size, device=device
            )
            iids = core.to_njt(xb)
            logits, extras = model(iids)
            logits_flat = logits.values()
            targets_flat = yb.view(-1)
            mask_flat = mb.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()

            l_ratio = sum([e.loss_ratio for e in extras], torch.tensor(0.0, device=device))
            loss = loss + l_ratio

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"Saving weights to {args.weights}")
        torch.save(model.state_dict(), args.weights)

    # generate from the model
    start = time.time()
    output = generate(
        model,
        max_new_tokens=args.max_new,
        prompt_len=args.prompt_len,
        temp=args.temp,
        confidence_threshold=args.confidence,
        top_k=args.top_k,
    )
    print(f"Total generation time: {time.time() - start:.2f} seconds")
    print(f"\nOutput:\n{output}")
