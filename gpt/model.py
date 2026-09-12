"""A small GPT — INCOMPLETE ON PURPOSE.

Four TODOs. Each one is a piece of the actual mechanism; the scaffolding around
them is done. Fill them in and `train.py` runs.

Shape convention used throughout, and worth committing to memory:

    B  batch size          how many independent sequences at once
    T  time / block_size   how many tokens of context
    C  channels / n_embd   the width of the residual stream
    H  n_head              number of attention heads
    hs C // H              head size

If you only track one thing while reading this file, track the shape of the
tensor flowing down the residual stream. It is (B, T, C) from the moment the
embeddings are added until the final layernorm, and every block writes back
into that same shape. The residual stream is the spine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False  # biases in the linear layers; off is slightly better and faster


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    The one idea: every position builds a query ("what am I looking for?"),
    a key ("what do I offer?") and a value ("what do I pass along if chosen?").
    Attention scores are query-dot-key for every pair of positions, softmaxed
    into weights, and used to take a weighted average of the values.

    Causal means position t may only attend to positions <= t. Without that,
    the model sees the answer it's being asked to predict and learns nothing.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # One fused projection producing q, k and v together, then split.
        # Three separate Linears would be mathematically identical but slower.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Lower-triangular matrix of ones — the causal mask. Registered as a
        # buffer so it moves to the GPU with .to(device) but isn't a parameter.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # TODO 1 — attention.
        #
        # a) Project and split:
        #        qkv = self.c_attn(x)          -> (B, T, 3C)
        #        q, k, v = qkv.split(C, dim=2) -> three of (B, T, C)
        #
        # b) Reshape each into heads and move the head dim next to batch, so
        #    every head is an independent attention problem:
        #        q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        #    giving (B, H, T, hs).
        #
        # c) Scores: q @ k.transpose(-2, -1) -> (B, H, T, T).
        #    Scale by 1/sqrt(hs). Without the scale, dot products grow with
        #    head size, softmax saturates, and gradients vanish. This is the
        #    "scaled" in scaled dot-product attention and it is not optional.
        #
        # d) Mask: wherever self.tril[:, :, :T, :T] == 0, set the score to
        #    float('-inf') — use Tensor.masked_fill. -inf becomes exactly zero
        #    after softmax, so those positions contribute nothing.
        #
        # e) Softmax over the last dim, then self.attn_dropout.
        #
        # f) Weighted sum of values: att @ v -> (B, H, T, hs).
        #
        # g) Reassemble the heads: transpose(1, 2) then .contiguous().view(B, T, C).
        #    The .contiguous() is required — transpose only changes strides,
        #    and .view() needs a contiguous buffer. Omit it and you get a
        #    loud error that tells you exactly this.
        #
        # h) Return self.resid_dropout(self.c_proj(y)).
        #
        # Once it works, try replacing c-f with:
        #     y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
        #             dropout_p=self.dropout if self.training else 0)
        # Same math, but it dispatches to FlashAttention — fused, and it never
        # materialises the (B, H, T, T) score matrix. That matrix is why
        # attention cost grows with the square of context length, and why
        # avoiding it is the single biggest memory win available on a 12GB card.
        raise NotImplementedError("TODO 1: CausalSelfAttention.forward")


class MLP(nn.Module):
    """Position-wise feed-forward network.

    Attention moves information *between* positions. This moves it *within* a
    position. Alternating the two is the whole architecture.

    The 4x expansion is convention from the original transformer paper and has
    stuck: project up to 4*n_embd, apply a nonlinearity, project back down.
    Most of a transformer's parameters live here, not in attention.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # TODO 2 — build three layers:
        #   self.c_fc    : Linear(n_embd -> 4 * n_embd, bias=config.bias)
        #   self.c_proj  : Linear(4 * n_embd -> n_embd, bias=config.bias)
        #   self.dropout : Dropout(config.dropout)
        raise NotImplementedError("TODO 2: MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2 (cont) — c_fc, then F.gelu, then c_proj, then dropout.
        # GELU rather than ReLU: it's smooth near zero, so gradients don't
        # die for slightly-negative inputs the way they do with a hard cutoff.
        raise NotImplementedError("TODO 2: MLP.forward")


class Block(nn.Module):
    """One transformer block: attention, then feed-forward, both residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 3 — two residual connections:
        #     x = x + self.attn(self.ln_1(x))
        #     x = x + self.mlp(self.ln_2(x))
        #     return x
        #
        # Two things worth more than the two lines suggest.
        #
        # The `x +` is the residual stream. Each block *adds* its contribution
        # rather than replacing the signal, so gradients have an unbroken
        # additive path from the loss all the way back to the embeddings.
        # This is what makes deep stacks trainable at all.
        #
        # The norm goes *before* the sublayer, not after. Pre-norm is why
        # modern transformers train without a learning-rate warmup babysitting
        # them. The original paper put it after and it was much more fragile.
        raise NotImplementedError("TODO 3: Block.forward")


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: what does this symbol mean?
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Position embeddings: where in the sequence am I?
                # Attention is permutation-invariant on its own — it has no
                # inherent notion of order — so position must be injected.
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the embedding table and the output projection are the
        # same matrix. Reading "what does this token mean" and writing "how
        # much do I predict this token" are inverse operations, so sharing
        # the weights saves parameters and usually helps.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections — with n_layer blocks all adding
        # into the stream, variance would otherwise grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence of {T} exceeds block_size"

        pos = torch.arange(T, dtype=torch.long, device=idx.device)

        # TODO 4 — assemble the forward pass:
        #
        #   tok_emb = self.transformer.wte(idx)   -> (B, T, C)
        #   pos_emb = self.transformer.wpe(pos)   -> (T, C)
        #   x = self.transformer.drop(tok_emb + pos_emb)
        #
        # Note the broadcast: (B, T, C) + (T, C) adds the same positional
        # vector to every sequence in the batch. That's intended.
        #
        #   for block in self.transformer.h:
        #       x = block(x)
        #   x = self.transformer.ln_f(x)
        #   logits = self.lm_head(x)              -> (B, T, vocab_size)
        #
        # Then the loss, when targets are given:
        #
        #   loss = F.cross_entropy(
        #       logits.view(-1, logits.size(-1)),
        #       targets.view(-1),
        #   )
        #
        # The flattening matters: cross_entropy wants (N, classes) and (N,).
        # We're folding batch and time together into one big pile of
        # independent predictions — B*T of them per step. When targets is
        # None (generation), return (logits, None).
        raise NotImplementedError("TODO 4: GPT.forward")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Complete — nothing to do here.

        Worth reading anyway: this is the entire inference loop of every
        language model you have ever used. Predict a distribution over the
        next token, sample one, append it, feed the whole thing back in.
        There is no more to it than this.
        """
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens — the model cannot attend
            # further back than its position embeddings go.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            # Only the last position's prediction matters.
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
