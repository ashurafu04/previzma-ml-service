import json
import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    SimulationRequest,
    SimulationResponse,
)
from app.services.forecast_service import predict_forecast
from app.services.simulation_service import run_simulation

logger = logging.getLogger("uvicorn.error")

SENSITIVE_HEADERS = {"authorization", "cookie", "set-cookie", "x-api-key", "x-auth-token"}

app = FastAPI(
    title="Previzma ML Service",
    description="Stateless ML engine for Previzma sales forecasts and simulations.",
    version="0.1.0",
)


def redact_headers(headers) -> dict[str, str]:
    return {
        key: "[REDACTED]" if key.lower() in SENSITIVE_HEADERS else value
        for key, value in headers.items()
    }


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    raw_body = await request.body()
    body_text = raw_body.decode("utf-8", errors="replace")

    try:
        received_body = json.loads(body_text) if body_text else None
    except json.JSONDecodeError:
        received_body = body_text

    validation_errors = jsonable_encoder(exc.errors())
    request_metadata = {
        "method": request.method,
        "path": request.url.path,
        "contentType": request.headers.get("content-type"),
        "contentLength": request.headers.get("content-length"),
        "upgrade": request.headers.get("upgrade"),
        "bodyWasEmpty": not bool(raw_body),
        "headers": redact_headers(request.headers),
    }

    logger.warning(
        "Request validation failed: method=%s path=%s content_type=%s "
        "content_length=%s upgrade=%s errors=%s body=%s headers=%s",
        request.method,
        request.url.path,
        request.headers.get("content-type"),
        request.headers.get("content-length"),
        request.headers.get("upgrade"),
        validation_errors,
        body_text,
        request_metadata["headers"],
    )

    return JSONResponse(
        status_code=422,
        content={
            "message": "Request validation failed",
            "path": request.url.path,
            "requestMetadata": request_metadata,
            "validationErrors": validation_errors,
            "receivedBody": received_body,
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    return predict_forecast(request)


@app.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest) -> SimulationResponse:
    return run_simulation(request)
