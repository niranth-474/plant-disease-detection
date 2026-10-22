from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class TopPrediction(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=100)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=100)
    top_predictions: list[TopPrediction]
