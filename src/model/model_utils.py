import torch
import math
from torch import Tensor


@torch.no_grad()
def flash_atention_forward(
    Q: Tensor, K: Tensor, V: Tensor, block_size: int = 1024, causal=True
) -> Tensor:
    # Q shape: [..., T, D]
    assert Q.shape == K.shape == V.shape
    *batch_shape, seq_len, embed_dim = Q.shape

    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(embed_dim)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        Q_block = Q[..., q_start:q_end, :]
        Qb = q_end - q_start

        l = torch.zeros(
            (*batch_shape, Qb, 1), device=Q.device, dtype=torch.float32
        )  # running sum (exponential)    shape ...,T,1
        m = torch.full(
            (*batch_shape, Qb, 1), -float("inf"), device=Q.device, dtype=torch.float32
        )  # running max
        o = torch.zeros(
            (*batch_shape, Qb, embed_dim), device=Q.device, dtype=torch.float32
        )

        for kv_start in range(0, seq_len, block_size):
            kv_end = min(kv_start + block_size, seq_len)

            if causal and kv_start >= q_end:
                break

            K_block = K[..., kv_start:kv_end, :]
            V_block = V[..., kv_start:kv_end, :]

            S_block = (Q_block @ K_block.transpose(-1, -2)) * scale
            S_block = S_block.float()

            if causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device)
                kv_idx = torch.arange(kv_start, kv_end, device=Q.device)
                mask = kv_idx[None, :] > q_idx[:, None]  # [Qb, Kb]
                S_block = S_block.masked_fill(mask, -float("inf"))

            S_max = S_block.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, S_max)

            exp_m_delta = torch.exp(m - m_new)
            P = torch.exp(S_block - m_new)

            l = exp_m_delta * l + P.sum(dim=-1, keepdim=True)
            o = exp_m_delta * o + (P @ V_block.float())

            m = m_new

        O[..., q_start:q_end, :] = (o / l).to(Q.dtype)

    return O
