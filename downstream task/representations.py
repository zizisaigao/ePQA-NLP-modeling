from typing import Optional
import torch

def mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden: [B,T,D], attention_mask: [B,T]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B,T,1]
    summed = (hidden * mask).sum(dim=1)                   # [B,D]
    denom = mask.sum(dim=1).clamp(min=1.0)               # [B,1]
    return summed / denom

def eos_pooling(hidden: torch.Tensor, input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    # Take representation at first EOS (or last token if not found)
    B, T, D = hidden.shape
    out = torch.zeros((B, D), device=hidden.device, dtype=hidden.dtype)
    for i in range(B):
        eos_pos = (input_ids[i] == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            j = int(eos_pos[0].item())
        else:
            j = T - 1
        out[i] = hidden[i, j]
    return out
