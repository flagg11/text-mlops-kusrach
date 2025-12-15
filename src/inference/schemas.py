from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    comments: List[str]

class PredictResponse(BaseModel):
    predictions: List[str]
