"""Model loading and prediction helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from src.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "model.pkl"
TRANSFORMER_PATH = REPO_ROOT / "transformer.pkl"

_state: dict[str, object] = {"model": None, "transformer": None}


def load_artifacts(
    model_path: Path | None = None, transformer_path: Path | None = None
) -> None:
    """Load model and transformer into the module-level cache.

    Idempotent: subsequent calls re-use the loaded objects unless paths change.
    Raises RuntimeError when either artifact is missing.
    """
    mp = Path(model_path) if model_path else MODEL_PATH
    tp = Path(transformer_path) if transformer_path else TRANSFORMER_PATH

    if not mp.exists():
        logger.error("model artifact missing", extra={"path": str(mp)})
        raise RuntimeError(f"Model artifact not found at {mp}")
    if not tp.exists():
        logger.error("transformer artifact missing", extra={"path": str(tp)})
        raise RuntimeError(f"Transformer artifact not found at {tp}")

    logger.info("loading artifacts", extra={"model": str(mp), "transformer": str(tp)})
    _state["model"] = joblib.load(mp)
    _state["transformer"] = joblib.load(tp)
    logger.info(
        "artifacts loaded",
        extra={"model_type": type(_state["model"]).__name__},
    )


def is_loaded() -> bool:
    return _state["model"] is not None and _state["transformer"] is not None


def reset() -> None:
    """Clear the cache (used by tests)."""
    _state["model"] = None
    _state["transformer"] = None


def _hash_features(features: PredictRequest) -> str:
    payload = json.dumps(features.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def predict(features: PredictRequest) -> PredictResponse:
    """Run a single prediction.

    Logs feature hash, prediction, probability, and latency.
    """
    if not is_loaded():
        logger.warning("predict called before artifacts loaded; loading now")
        load_artifacts()

    model = _state["model"]
    transformer = _state["transformer"]

    feature_hash = _hash_features(features)
    started = time.perf_counter()

    df = pd.DataFrame([features.model_dump()])
    transformed = transformer.transform(df)
    transformed = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())

    prediction = int(model.predict(transformed)[0])
    proba = float(model.predict_proba(transformed)[0][1])

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "prediction served",
        extra={
            "event": "prediction_served",
            "feature_hash": feature_hash,
            "prediction": prediction,
            "probability": round(proba, 4),
            "latency_ms": round(latency_ms, 2),
        },
    )

    return PredictResponse(prediction=prediction, probability=proba)
