import importlib.util
import os
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

def _import_from_path(py_path: str) -> ModuleType:
    py_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location("user_model_module", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from path: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def build_model_from_user_code(model_py: str, model_class: str, model_kwargs: Dict[str, Any]) -> nn.Module:
    mod = _import_from_path(model_py)
    if not hasattr(mod, model_class):
        raise AttributeError(f"Model class '{model_class}' not found in {model_py}. Available: {dir(mod)}")
    cls = getattr(mod, model_class)
    model = cls(**model_kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{model_class} is not a torch.nn.Module")
    return model

def load_lm(
    ckpt_path: str,
    device: torch.device,
    model_py: Optional[str] = None,
    model_class: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Case 1: whole model was saved
    if isinstance(ckpt, nn.Module):
        model = ckpt
        model.to(device)
        return model

    # Case 2: dict checkpoint
    if isinstance(ckpt, dict):
        # If it contains a pickled model
        for k in ["model", "lm", "net"]:
            if k in ckpt and isinstance(ckpt[k], nn.Module):
                model = ckpt[k]
                model.to(device)
                return model

        # state dict under common keys
        state = None
        for k in ["model_state", "model_state_dict", "state_dict", "lm_state", "net_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # ckpt itself is a state_dict-like dict
            state = ckpt

        if state is None:
            raise ValueError(f"Could not find model state dict in checkpoint keys={list(ckpt.keys())}")

        # Need to construct model from user code, unless user saved enough to reconstruct automatically
        if model_py and model_class:
            kwargs = model_kwargs or {}
            model = build_model_from_user_code(model_py, model_class, kwargs)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[lm_loader] Warning: missing keys (showing up to 20): {missing[:20]}")
            if unexpected:
                print(f"[lm_loader] Warning: unexpected keys (showing up to 20): {unexpected[:20]}")
            model.to(device)
            return model

        raise ValueError(
            "Checkpoint contains only a state_dict but no model object. "
            "Provide --model_py and --model_class (and optionally --model_kwargs_json) to construct the model."
        )

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

def get_hidden_states(model: nn.Module, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Tries multiple conventions to extract last hidden states [B,T,D].
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        out = model.encode(input_ids, attention_mask)  # type: ignore
        if isinstance(out, torch.Tensor):
            return out

    if hasattr(model, "get_hidden_states") and callable(getattr(model, "get_hidden_states")):
        out = model.get_hidden_states(input_ids, attention_mask)  # type: ignore
        if isinstance(out, torch.Tensor):
            return out

    #out = model(input_ids, attention_mask) if attention_mask is not None else model(input_ids)
    try:
        out = model(input_ids, attention_mask) if attention_mask is not None else model(input_ids)
    except TypeError:
        out = model(input_ids)


    # HuggingFace-like output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state  # type: ignore

    # dict output
    if isinstance(out, dict):
        for k in ["hidden_states", "last_hidden", "h", "x"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        # sometimes logits + hidden
        if "logits" in out and "hidden" in out and isinstance(out["hidden"], torch.Tensor):
            return out["hidden"]

    # tuple output
    if isinstance(out, (tuple, list)):
        # common patterns: (logits, hidden) or (hidden, ...)
        for item in out:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
        # as fallback, try second element
        if len(out) >= 2 and isinstance(out[1], torch.Tensor) and out[1].dim() == 3:
            return out[1]

    raise ValueError(
        "Unable to extract hidden states from model output. "
        "Consider adding an encode() method to your Transformer that returns [B,T,D]."
    )
