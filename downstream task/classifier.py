from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from lm_loader import get_hidden_states
from representations import mean_pooling, eos_pooling

PoolType = Literal["mean", "eos"]

@dataclass
class TwoStageOutput:
    logits_stage0: torch.Tensor  # [B,2] for 0 vs {1,2}
    logits_stage12: torch.Tensor # [B,2] for 1 vs 2 (meaningful for non-0 region)
    pred_3class: torch.Tensor    # [B] in {0,1,2}

class TwoStageClassifier(nn.Module):
    def __init__(self, lm: nn.Module, hidden_dim: int, pool: PoolType = "mean", eos_id: Optional[int] = None):
        super().__init__()
        self.lm = lm
        self.pool = pool
        self.eos_id = eos_id
        self.head0 = nn.Linear(hidden_dim, 2)   # 0 vs {1,2}
        self.head12 = nn.Linear(hidden_dim, 2)  # 1 vs 2

    def _pool(self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return mean_pooling(hidden, attention_mask)
        if self.pool == "eos":
            if self.eos_id is None:
                raise ValueError("eos_id must be provided for eos pooling")
            return eos_pooling(hidden, input_ids, self.eos_id)
        raise ValueError(f"Unknown pool type: {self.pool}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TwoStageOutput:
        hidden = get_hidden_states(self.lm, input_ids, attention_mask)  # [B,T,D]
        z = self._pool(hidden, input_ids, attention_mask)               # [B,D]
        logits0 = self.head0(z)
        logits12 = self.head12(z)

        pred0 = torch.argmax(logits0, dim=-1)  # 0 means label==0, 1 means label in {1,2}
        pred12 = torch.argmax(logits12, dim=-1)  # 0 means label==1, 1 means label==2

        pred3 = torch.where(pred0 == 0, torch.zeros_like(pred0), pred12 + 1)
        return TwoStageOutput(logits_stage0=logits0, logits_stage12=logits12, pred_3class=pred3)
