"""Inference module for loan predictions."""
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.feature_engineering import FeatureEngineer

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)


class LoanInference:
    """Class responsible for model inference."""

    def __init__(self, model_path: Any):
        self.model_path = Path(model_path)
        logger.info("Initializing LoanInference with model path: %s", self.model_path)

        if not self.model_path.exists():
            logger.error("Model artifact not found at %s", self.model_path)
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        try:
            artifact = joblib.load(self.model_path)
            self.model = artifact.get("model")
            self.scaler = artifact.get("scaler")
            self.metrics = artifact.get("metrics", {}) or {}

            if self.model is None or self.scaler is None:
                logger.error(
                    "Model artifact is missing required components ('model' and 'scaler')."
                )
                raise ValueError(
                    "Model artifact is missing required components ('model' and 'scaler')."
                )

            logger.info("Successfully loaded model artifact.")
        except Exception as exc:
            logger.error("Failed to load model artifact from %s: %s", self.model_path, exc)
            raise RuntimeError(
                f"Failed to load model artifact from {self.model_path}: {exc}"
            ) from exc

    def predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        credit_score: int,
        annual_income: int,
        loan_amount: int,
        dti_ratio: float,
        employment_tenure: int,
    ) -> Dict[str, Any]:
        """Predict default probability for a single applicant profile."""
        logger.info("Received prediction request.")
        try:
            row = pd.DataFrame(
                [
                    [
                        credit_score,
                        annual_income,
                        loan_amount,
                        dti_ratio,
                        employment_tenure,
                    ]
                ],
                columns=FeatureEngineer.FEATURES,
            )

            scaled = self.scaler.transform(row)
            prob = float(self.model.predict_proba(scaled)[0, 1])
            is_default = int(prob >= 0.5)

            result = {
                "default_prediction": is_default,
                "default_probability": round(prob, 4),
                "risk_status": (
                    "High Risk (Review/Reject Adverse Status)"
                    if is_default
                    else "Low Risk (Auto-Approve)"
                ),
            }
            logger.info("Prediction successful: %s", result)
            return result
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise RuntimeError(f"Failed to predict: {exc}") from exc
