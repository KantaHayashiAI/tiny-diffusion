# H-Net (Autoregressive) training/inference on Tiny Shakespeare
import argparse
import os
import time
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from hnet_impl import HNetLM
import hnet_core as core

# -----------------
# hyperparameters
# -----------------
batch_size = 16
block_size = 256
max_iters = 2000
learning_rate = 3e-4
eval_interval = 200
eval_iters = 50
# --------------


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x]


@torch.no_grad()
def generate(model, max_new_tokens, prompt_len=16, temp=0.8):
    device = next(model.parameters()).device
    vocab = core.vocab_ar
    amp_ctx = (
        torch.autocast("cuda", torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    # Start with first prompt_len tokens from data as context
    x = core.train_data_ar[:prompt_len].unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        cur_context = x[:, -block_size:]
        iids = core.to_njt(cur_context)
        with amp_ctx:
            logits, _extra = model(iids)
        logits = core.flat_logits_to_btt(
            logits, cur_context.size(0), cur_context.size(1), vocab.vocab_size
        )
        logits = logits[:, -1, :]
        probs = F.softmax(logits / temp, dim=-1)
        next_token = (
            torch.argmax(probs, dim=-1, keepdim=True)
            if temp == 0
            else torch.multinomial(probs, num_samples=1)
        )
        x = torch.cat((x, next_token), dim=1)

    return vocab.decode(x[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H-Net autoregressive LM")
    parser.add_argument("--train", action="store_true", help="Train from scratch")
    parser.add_argument(
        "--arch",
        type=str,
        default="m2,T2",
        help="Stage arch list, e.g. 'm2,T2' or 'T2,T2'",
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
        default="weights/hnet_gpt.pt",
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

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("H-Net requires CUDA (flash-attn/mamba-ssm kernels).")

    arch = [a.strip() for a in args.arch.split(",") if a.strip()]
    d_model = parse_int_list(args.d_model)
    if len(arch) != len(d_model):
        raise ValueError("--arch and --d-model must have the same number of stages")

    # Build config
    c = core.build_hnet_config(
        vocab_size=core.vocab_ar.vocab_size,
        d_model=d_model,
        arch=arch,
        is_causal=True,
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
                losses = core.estimate_loss_ar(
                    model,
                    eval_iters=eval_iters,
                    batch_size=batch_size,
                    block_size=block_size,
                )
                print(
                    f"step {iter}: train loss {losses['train']:.4f},"
                    f"val loss {losses['val']:.4f}, time {time.time() - start:.2f} seconds"
                )
                sample = generate(
                    model, max_new_tokens=240, prompt_len=args.prompt_len
                )
                print(f"Sample:\n{sample}\n")

            xb, yb = core.get_batch_ar(
                "train", batch_size=batch_size, block_size=block_size, device=device
            )
            iids = core.to_njt(xb)
            lbls = core.to_njt(yb).long()
            amp_ctx = (
                torch.autocast("cuda", torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                (l_avg, _l_sum), extras = model(iids, lbls)
            l_ratio = sum([e.loss_ratio for e in extras], torch.tensor(0.0, device=device))
            loss = l_avg + l_ratio

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"Total training time: {time.time() - start:.2f} seconds")
        print(f"Saving weights to {args.weights}")
        torch.save(model.state_dict(), args.weights)

    # generate from the model
    start = time.time()
    output = generate(
        model, max_new_tokens=args.max_new, prompt_len=args.prompt_len, temp=args.temp
    )
    print(f"Total generation time: {time.time() - start:.2f} seconds")
    print(f"\nOutput:\n{output}")
