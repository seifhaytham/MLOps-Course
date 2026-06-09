"""Litestar application: home, health, and predict endpoints."""

from __future__ import annotations

import logging
import logging.config
from typing import Any

from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig
from litestar.di import Provide
from litestar.logging import LoggingConfig
from litestar.status_codes import HTTP_200_OK

from src.predictor import is_loaded, load_artifacts, predict
from src.schemas import (
    HealthResponse,
    HomeResponse,
    PredictRequest,
    PredictResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging_config = LoggingConfig(
    root={"level": "INFO", "handlers": ["queue_listener"]},
    formatters={
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    log_exceptions="always",
)

logger = logging.getLogger(__name__)

# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────

def on_startup() -> None:
    """Load artifacts once at startup so the first request is not cold."""
    logger.info("startup: loading artifacts")
    load_artifacts()
    logger.info("startup: artifacts ready")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@get("/", status_code=HTTP_200_OK, tags=["meta"])
async def home() -> HomeResponse:
    """Service info and link to docs."""
    return HomeResponse(
        service="churn-prediction-api",
        version="0.1.0",
        docs="/schema/swagger",
    )


@get("/health", status_code=HTTP_200_OK, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness + readiness probe."""
    loaded = is_loaded()
    logger.info("health check", extra={"model_loaded": loaded})
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )


@post("/predict", status_code=HTTP_200_OK, tags=["prediction"])
async def predict_endpoint(data: PredictRequest) -> PredictResponse:
    """Run churn prediction for a single customer record."""
    logger.info(
        "predict request received",
        extra={"geography": data.Geography, "age": data.Age},
    )
    return predict(data)


# ── Application factory ───────────────────────────────────────────────────────

app = Litestar(
    route_handlers=[home, health, predict_endpoint],
    on_startup=[on_startup],
    logging_config=logging_config,
)