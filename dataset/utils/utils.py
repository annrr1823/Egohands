from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import torch


def _to_torch(x: Any, dtype: Optional[torch.dtype] = None) -> Any:
    """
    Convert common types to torch.Tensor while preserving dict and list structure.
    """

    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype=dtype)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t if dtype is None else t.to(dtype=dtype)
    if isinstance(x, (float, int)):
        t = torch.tensor(x)
        return t if dtype is None else t.to(dtype=dtype)
    
    if isinstance(x, dict):
        return {k: _to_torch(v, dtype=dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_torch(v, dtype=dtype) for v in x)
    
    return x

def _temporal_stack(items: List[Any]) -> Any:
    """
    Stack a list of items along time dimension (dim=0).
    Supports:
      - torch.Tensor: stack -> [T, ...]
      - dict[str, torch.Tensor]: stack each field -> [T, ...]
      - None: return None
    """
    if len(items) == 0:
        return None
    if items[0] is None:
        return None

    item = items[0]
    if isinstance(item, torch.Tensor):
        return torch.stack(items, dim=0)

    if isinstance(item, dict):
        out: Dict[str, Any] = {}
        for k in item.keys():
            out[k] = _temporal_stack([it[k] for it in items])
        return out

    return items

def _pad_clip(clip: List[dict], target_len: int, pad_mode: Literal["repeat"]) -> List[dict]:
    if len(clip) >= target_len:
        return clip[:target_len]
    
    if len(clip) == 0:
        return [{} for _ in range(target_len)]

    if pad_mode == "repeat":
        last = clip[-1]
        return clip + [last for _ in range(target_len - len(clip))]

    raise ValueError(f"Unknown pad_mode: {pad_mode}")