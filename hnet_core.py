"""
Shared utilities for H-Net training/inference on the tiny Shakespeare dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nested
from torch.nn import functional as F

from hnet_impl import HNetConfig

# -----------------
# Dataset and vocab
# -----------------

with open("data.txt", "r", encoding="utf-8") as f:
    _text = f.read()


@dataclass(frozen=True)
class CharVocab:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]
    vocab_size: int
    mask_token: str | None = None
    mask_token_id: int | None = None

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def _choose_mask_token(chars: list[str]) -> str:
    for candidate in ["_", "\x00", "\x01", "\x02"]:
        if candidate not in chars:
            return candidate
    raise RuntimeError("Could not find a free mask token character.")


def build_vocab(text: str, *, add_mask: bool = False) -> CharVocab:
    chars = sorted(list(set(text)))
    mask_token = None
    mask_token_id = None
    if add_mask:
        mask_token = _choose_mask_token(chars)
        chars = [mask_token] + chars
        mask_token_id = 0

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return CharVocab(
        chars=chars,
        stoi=stoi,
        itos=itos,
        vocab_size=len(chars),
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


vocab_ar = build_vocab(_text, add_mask=False)
vocab_diff = build_vocab(_text, add_mask=True)

# Encode full dataset
_data_ar = torch.tensor(vocab_ar.encode(_text), dtype=torch.long)
_data_diff = torch.tensor(vocab_diff.encode(_text), dtype=torch.long)

# Train/val splits
_n = int(0.9 * len(_data_ar))
train_data_ar = _data_ar[:_n]
val_data_ar = _data_ar[_n:]
train_data_diff = _data_diff[:_n]
val_data_diff = _data_diff[_n:]


# ---------------
# Batch utilities
# ---------------

def get_batch_ar(split: str, *, batch_size: int, block_size: int, device: str):
    data = train_data_ar if split == "train" else val_data_ar
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)


def get_batch_diffusion(
    split: str, *, batch_size: int, block_size: int, device: str
):
    data = train_data_diff if split == "train" else val_data_diff
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = x.clone()

    # Mask tokens with random probability per sample
    mask_probs = torch.rand(batch_size, 1)
    mask = torch.rand(batch_size, block_size) < mask_probs
    x[mask] = vocab_diff.mask_token_id

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask


def to_njt(batch: torch.Tensor):
    # Convert (B, T, ...) -> nested jagged with per-row tensors
    return nested.nested_tensor(list(batch.unbind(0)), layout=torch.jagged)


def flat_logits_to_btt(logits, batch_size: int, seq_len: int, vocab_size: int):
    # logits is a nested tensor (values are flat). For uniform lengths, reshape.
    return logits.values().view(batch_size, seq_len, vocab_size)


# ------------------
# H-Net config utils
# ------------------

def build_hnet_config(
    *,
    vocab_size: int,
    d_model: list[int],
    arch: list[str],
    is_causal: bool,
):
    c = HNetConfig.create_reasonable_config(D=d_model, arch=arch)
    c = replace(c, vocab_size=vocab_size)
    if not is_causal:
        attn_cfg = replace(c.attn_cfg, is_causal=[False] * len(c.d_model))
        c = replace(c, attn_cfg=attn_cfg)
    return c


@torch.no_grad()
def estimate_loss_ar(model, *, eval_iters: int, batch_size: int, block_size: int):
    out = {}
    device = next(model.parameters()).device
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_ar(
                split, batch_size=batch_size, block_size=block_size, device=device
            )
            iids = to_njt(X)
            lbls = to_njt(Y).long()
            (l_mean, _l_sum), _extra = model(iids, lbls)
            losses[k] = l_mean.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def estimate_loss_diffusion(
    model,
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    vocab_size: int,
):
    out = {}
    device = next(model.parameters()).device
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch_diffusion(
                split, batch_size=batch_size, block_size=block_size, device=device
            )
            iids = to_njt(X)
            logits, _extra = model(iids)
            logits_flat = logits.values()
            targets_flat = Y.view(-1)
            mask_flat = M.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            loss = (loss * mask_flat).sum() / mask_flat.sum()
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
