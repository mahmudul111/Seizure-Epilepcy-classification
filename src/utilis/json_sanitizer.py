from typing import Any
from pydantic import BaseModel
import numpy as np
import pandas as pd


def _sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        if isinstance(obj, BaseModel):
            return _sanitize(obj.dict())
    except Exception:
        pass

    if isinstance(obj, pd.DataFrame):
        return _sanitize(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _sanitize(obj.tolist())

    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return obj.tolist() if hasattr(obj, "tolist") else obj

    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]

    return obj
