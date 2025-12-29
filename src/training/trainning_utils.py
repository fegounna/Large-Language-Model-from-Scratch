import torch
from torch.optim import Optimizer
import math
from typing import Optional, Callable, Tuple, Iterable


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
        Parameters
    ----------
    logits : torch.Tensor
        Shape: (..., vocab_size)
        Batch-like dimensions come first.
    targets : torch.Tensor
        Shape: (...)
        Integer class indices (xi+1).
    """

    log_softmax = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -log_softmax.gather(dim=-1, index=targets.unsqueeze(-1))
    return loss.squeeze(-1).mean()


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1

                t = state["t"]
                m = state["m"]
                v = state["v"]

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                state["m"] = m
                state["v"] = v

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data -= lr * weight_decay * p
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

        return loss


def cosine_annealing_lr_schedule(
    t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int
) -> float:
    if t < T_w:
        return t / T_w * alpha_max
    if t > T_c:
        return alpha_min

    return alpha_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (
        alpha_max - alpha_min
    )


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    total_norm_sq = 0.0
    for g in grads:
        total_norm_sq += g.data.norm(2).item() ** 2

    total_norm = total_norm_sq**0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.data.mul_(scale)
