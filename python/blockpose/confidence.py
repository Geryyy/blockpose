"""Synthetic-only validity confidence calibration utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

FEATURE_SCHEMA = (
    "support",
    "height_consistency",
    "visible_face_coverage",
    "free_space_consistency",
)
ARTIFACT_VERSION = 1
# Deliberately absent until a selected-output corpus contains enough organic
# positives and negatives. Experimental artifacts must be passed explicitly.
DEFAULT_ARTIFACT = Path(__file__).with_name("selected_output_calibration.json")


@dataclass(frozen=True)
class CalibrationArtifact:
    version: int
    schema: list[str]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float
    l2: float
    development_seeds: list[int]
    gate_seeds: list[int]

    def validate(self) -> None:
        if self.version != ARTIFACT_VERSION:
            raise ValueError(f"unsupported calibration artifact version: {self.version}")
        if tuple(self.schema) != FEATURE_SCHEMA:
            raise ValueError("calibration feature schema does not match runtime schema")
        width = len(FEATURE_SCHEMA)
        if not all(len(values) == width for values in (self.means, self.scales, self.coefficients)):
            raise ValueError("calibration artifact vectors have the wrong length")
        values = np.array([*self.means, *self.scales, *self.coefficients, self.intercept])
        if not np.all(np.isfinite(values)) or np.any(np.asarray(self.scales) <= 0.0):
            raise ValueError("calibration artifact contains invalid numeric values")
        if not np.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("calibration artifact l2 must be finite and non-negative")


def feature_vector(features: dict[str, float]) -> np.ndarray:
    if tuple(features) != FEATURE_SCHEMA:
        raise ValueError("diagnostic features do not match fixed calibration schema/order")
    vector = np.array([features[name] for name in FEATURE_SCHEMA], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("calibration features must be finite")
    return vector


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1.0,
    development_seeds: list[int],
    gate_seeds: list[int],
) -> CalibrationArtifact:
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and non-negative")
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_SCHEMA):
        raise ValueError("matrix has wrong calibration feature width")
    if labels.shape != (len(matrix),) or set(np.unique(labels)) - {0.0, 1.0}:
        raise ValueError("labels must be aligned binary values")
    if not ({0.0, 1.0} <= set(labels)):
        raise ValueError("calibration requires both positive and negative examples")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    normalized = (matrix - means) / scales

    def objective(theta):
        logits = normalized @ theta[:-1] + theta[-1]
        loss = np.logaddexp(0.0, logits).sum() - labels @ logits
        loss += 0.5 * l2 * np.dot(theta[:-1], theta[:-1])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        gradient = np.r_[normalized.T @ (probabilities - labels) + l2 * theta[:-1],
                         np.sum(probabilities - labels)]
        return float(loss), gradient

    result = minimize(objective, np.zeros(matrix.shape[1] + 1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"calibration optimization failed: {result.message}")
    artifact = CalibrationArtifact(
        ARTIFACT_VERSION,
        list(FEATURE_SCHEMA),
        means.tolist(),
        scales.tolist(),
        result.x[:-1].tolist(),
        float(result.x[-1]),
        float(l2),
        list(development_seeds),
        list(gate_seeds),
    )
    artifact.validate()
    return artifact


def predict(features: dict[str, float], artifact: CalibrationArtifact | None) -> float:
    """Return calibrated validity, or conservative zero when unavailable/invalid."""
    if artifact is None:
        return 0.0
    try:
        artifact.validate()
        vector = feature_vector(features)
    except (TypeError, ValueError):
        return 0.0
    normalized = (vector - artifact.means) / artifact.scales
    logit = float(normalized @ artifact.coefficients + artifact.intercept)
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))


def save_artifact(artifact: CalibrationArtifact, path: Path) -> None:
    artifact.validate()
    path.write_text(json.dumps(asdict(artifact), indent=2) + "\n")


def load_artifact(path: Path = DEFAULT_ARTIFACT) -> CalibrationArtifact | None:
    try:
        artifact = CalibrationArtifact(**json.loads(path.read_text()))
        artifact.validate()
        return artifact
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=1)
def runtime_artifact() -> CalibrationArtifact | None:
    return load_artifact()


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if labels.shape != probabilities.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("metrics require aligned non-empty vectors")
    if not np.all(np.isfinite(labels)) or set(np.unique(labels)) - {0.0, 1.0}:
        raise ValueError("metric labels must be finite binary values")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("metric probabilities must be finite values in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        count = int(mask.sum())
        confidence = float(probabilities[mask].mean()) if count else None
        accuracy = float(labels[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(confidence - accuracy)
        reliability.append({"low": float(edges[index]), "high": float(edges[index + 1]),
                            "count": count, "confidence": confidence, "accuracy": accuracy})
    selected = probabilities >= 0.8
    return {
        "count": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": float(ece),
        "precision_at_0_8": float(labels[selected].mean()) if selected.any() else None,
        "coverage_at_0_8": float(selected.mean()),
        "selected_at_0_8": int(selected.sum()),
        "reliability": reliability,
    }
