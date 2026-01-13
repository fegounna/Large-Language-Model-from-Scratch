"""Decoder-only Transformer language with RoPE, RMSNorm, and KV caching."""

import torch
import torch.nn as nn
import math
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )  # W is stored as row major like in C

        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (..., in_features)
        W shape: (out_features, in_features)
         output: (..., out_features)"""
        return x @ self.W.t()


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        return (xf / rms * self.weight).type_as(x)


def _round_up_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        if d_ff is None:
            d_ff_raw = int(math.floor((8.0 / 3.0) * d_model))
            self.d_ff = _round_up_to_multiple(d_ff_raw, 64)

        self.w_gate = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w_up = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w_down = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(x)
        gate = g * torch.sigmoid(g)
        u = self.w_up(x)
        hidden = gate * u

        return self.w_down(hidden)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = int(head_dim)
        self.max_seq_len = max_seq_len

        idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (idx / head_dim))  # [D/2]

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        angles = torch.einsum("s,d->sd", positions, inv_freq)

        self.register_buffer("cos", torch.cos(angles).to(dtype=dtype), persistent=False)
        self.register_buffer("sin", torch.sin(angles).to(dtype=dtype), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """x : batch,number_heads,seq_len,dim_head
        start_pos is used when we we are using kv caching
        """

        assert start_pos + x.shape[-2] <= self.max_seq_len, (
            f"RoPE cache too small: need positions up to {start_pos + x.shape[-2]}, "
            f"but max_seq_len={self.max_seq_len}"
        )

        seq_len = x.shape[-2]
        cos = self.cos[start_pos : start_pos + seq_len].view(1, 1, seq_len, -1)

        sin = self.sin[start_pos : start_pos + seq_len].view(1, 1, seq_len, -1)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_new = torch.empty_like(x)

        x_new[..., 0::2] = x_even * cos - x_odd * sin
        x_new[..., 1::2] = x_even * sin + cos * x_odd

        return x_new


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)

    return attn @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len_seq: int,
        num_kv_heads: int | None = None,
        causal: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )
        self.num_groups = num_heads // self.num_kv_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads
        self.max_len_seq = max_len_seq

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(
            d_model, self.num_kv_heads * self.dim_head, device=device, dtype=dtype
        )
        self.w_v = Linear(
            d_model, self.num_kv_heads * self.dim_head, device=device, dtype=dtype
        )
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.causal = causal
        self.rope = RotaryPositionalEmbedding(
            theta=10000.0,
            dim_head=self.dim_head,
            max_seq_len=max_len_seq,
            device=device,
            dtype=dtype,
        )
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(
                    torch.ones(
                        (max_len_seq, max_len_seq), dtype=torch.bool, device=device
                    )
                ),
                persistent=False,
            )  # 1 means attend

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_cache: bool = False,  # kv caching for causal inference
    ) -> torch.Tensor:
        """
        x: (b, T_new, d) if use_cache else (b, T, d)
        padding_mask: (B, T_new) if use_cache else (B, T) bool where True=real token, False=pad
        """

        q = self.w_q(x)

        q = rearrange(
            q,
            "batch len_seq (n_heads d_head) -> batch n_heads len_seq d_head",
            n_heads=self.num_heads,
            d_head=self.dim_head,
        )

        if use_cache:
            k_new = self.w_k(x)
            v_new = self.w_v(x)
            k_new = rearrange(
                k_new,
                "batch len_seq (n_heads d_head) -> batch n_heads len_seq d_head",
                n_heads=self.num_kv_heads,
                d_head=self.dim_head,
            )
            v_new = rearrange(
                v_new,
                "batch len_seq (n_heads d_head) -> batch n_heads len_seq d_head",
                n_heads=self.num_kv_heads,
                d_head=self.dim_head,
            )

            if self.cache_k is not None:
                starting_pos = self.cache_k.shape[-2]
                k_new = self.rope(k_new, start_pos=starting_pos)

                self.cache_k = torch.concat(
                    [self.cache_k, k_new], dim=-2
                )  # This could lead to memory fragmentation (one could pre allocate memory to cache)
                self.cache_v = torch.concat([self.cache_v, v_new], dim=-2)
            else:
                starting_pos = 0
                k_new = self.rope(k_new, start_pos=starting_pos)
                self.cache_k, self.cache_v = k_new, v_new
            k, v = self.cache_k, self.cache_v
            q = self.rope(q, start_pos=starting_pos)

        else:
            k = self.w_k(x)
            v = self.w_v(x)

            k = rearrange(
                k,
                "batch len_seq (n_heads d_head) -> batch n_heads len_seq d_head",
                n_heads=self.num_kv_heads,
                d_head=self.dim_head,
            )
            v = rearrange(
                v,
                "batch len_seq (n_heads d_head) -> batch n_heads len_seq d_head",
                n_heads=self.num_kv_heads,
                d_head=self.dim_head,
            )

            q = self.rope(q)
            k = self.rope(k)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_groups, dim=-3)
            v = v.repeat_interleave(self.num_groups, dim=-3)

        attn_mask = None

        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            attn_mask = padding_mask[:, None, None, :]

        if self.causal:
            Tq = q.shape[-2]
            Tk = k.shape[-2]

            if use_cache:
                causal_mask = self.causal_mask[Tk - Tq : Tk, :Tk]
            else:
                causal_mask = self.causal_mask[:Tq, :Tk]

            causal_mask = causal_mask[None, None, :, :]  # (1,1,Tq,Tk)
            attn_mask = causal_mask if attn_mask is None else (attn_mask & causal_mask)

        # TO use Flash Attention
        # out = nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,is_causal=self.causal)
        out = scaled_dot_attention(q, k, v, mask=attn_mask)  # B h len_seq d_head

        out = rearrange(
            out, "batch n_heads len_seq d_head -> batch len_seq (n_heads d_head)"
        )

        return self.w_o(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len_seq: int,
        num_kv_heads: int | None = None,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff

        self.norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.causal_mha = MultiHeadSelfAttention(
            d_model,
            num_heads,
            num_kv_heads=num_kv_heads,
            max_len_seq=max_len_seq,
            causal=True,
            device=device,
            dtype=dtype,
        )
        self.norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x_norm1 = self.norm_1(x)
        x_att = self.causal_mha(x_norm1, padding_mask=padding_mask, use_cache=use_cache)
        x_res = x + x_att
        x_norm2 = self.norm_2(x_res)
        x_ff = self.ff(x_norm2)
        x_out = x_res + x_ff
        return x_out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        d_ff: int | None = None,
        device=None,
        dtype=None,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff

        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    max_len_seq=max_seq_len,
                    d_ff=d_ff,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if tie_weights:
            self.lm_head.W = self.embedding.weight

    def reset_cache(self):
        for layer in self.layers:
            layer.causal_mha.reset_cache()

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        token_ids: (B, T)
        padding_mask: (B, T) bool where True=real token, False=pad
        Returns: logits (B, T, vocab_size)
        """
        x = self.embedding(token_ids)

        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, use_cache=use_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits
