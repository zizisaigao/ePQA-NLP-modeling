import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

@dataclass
class EarlyStopper:
    patience: int = 3
    mode: str = "max"  # "max" for metric like F1, "min" for loss
    best: Optional[float] = None
    bad_epochs: int = 0
    best_state: Optional[Dict[str, Any]] = None

    def step(self, value: float, state: Dict[str, Any]) -> bool:
        improved = False
        if self.best is None:
            improved = True
        else:
            if self.mode == "max":
                improved = value > self.best
            else:
                improved = value < self.best

        if improved:
            self.best = value
            self.bad_epochs = 0
            # store CPU state dicts for portability
            self.best_state = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in state.items()}
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
