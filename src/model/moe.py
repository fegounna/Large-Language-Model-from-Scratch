import torch.nn as nn
import torch


class MoE(nn.Module):
    def __init__(
        self, d_model: int, n_experts: int, top_k: int, bias_gate: bool = False
    ):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, n_experts, bias=bias_gate)

        self.We = nn.Parameter(torch.empty(n_experts, d_model, d_model))
        nn.init.xavier_uniform_(self.We)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        assert D == self.d_model

        logits = self.gate(x)  # (B,T,E)
        
        top_val, top_idx = torch.topk(logits,k=self.top_k,dim=-1) # both (B,T,K)

        top_w = nn.functional.softmax(top_val,dim=-1)  # (B,T,K)

        W_selected = self.We[top_idx] # (B,T,K,D,D)

        x_experts = torch.einsum("btd,btkdm->btkm",x,W_selected)

        out = (x_experts * top_w.unsqueeze(-1)).sum(dim=2)

        return out
