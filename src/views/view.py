from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from utilis.load_and_infer import run_inference
# from utilis.json_sanitizer import _sanitize
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


router_v1 = APIRouter(tags=["v1"])


class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True


class PredictionRequest(BaseModel):
    modelDirectory: str = Field(
        default="extracted_models/saved_models", description="Path to the directory containing the saved model files."
    )
    edfFilePath: str = Field(default="data/chb01_03.edf", description="Path to the EDF file for inference.")


@router_v1.get("/")
def read_root() -> dict[str, str]:
    try:
        return {"message": "Welcome to FastAPI!"}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@router_v1.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@router_v1.post("/prediction")
def make_prediction(data: PredictionRequest) -> dict[str, Any]:
    try:
        prediction, _, _s, model_result = run_inference(data.modelDirectory, data.edfFilePath)

        san_input = _sanitize(data)
        san_prediction = _sanitize(prediction)
        san_model_result = _sanitize(model_result)

        enc_input = jsonable_encoder(san_input)
        enc_prediction = jsonable_encoder(san_prediction)
        enc_model_result = jsonable_encoder(san_model_result)

        return {"model_result": enc_model_result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
