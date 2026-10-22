import logging
import pickle
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.schemas import HealthResponse, PredictionResponse
from backend.services.prediction_service import PredictionService


logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        application.state.prediction_service = PredictionService.from_files()
    except (
        EOFError,
        FileNotFoundError,
        OSError,
        pickle.UnpicklingError,
        ValueError,
        RuntimeError,
    ):
        logger.exception("Unable to load the plant disease prediction model.")
        application.state.prediction_service = None
    yield


app = FastAPI(
    title="Plant Disease Detection API",
    description="Predict plant leaf disease classes from an uploaded image.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def health_response() -> HealthResponse:
    is_loaded = getattr(app.state, "prediction_service", None) is not None
    return HealthResponse(
        status="ok" if is_loaded else "model_unavailable",
        model_loaded=is_loaded,
    )


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return health_response()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return health_response()


@app.get("/api/v1/health", response_model=HealthResponse)
def versioned_health() -> HealthResponse:
    return health_response()


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    service: PredictionService | None = getattr(
        app.state, "prediction_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable. Check the server configuration.",
        )

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the supported limit.",
                )
            image.load()
            decoded_image = image.copy()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as error:
        raise HTTPException(
            status_code=415,
            detail="The uploaded file is not a supported image.",
        ) from error

    result = service.predict(decoded_image)
    return PredictionResponse.model_validate(result)
