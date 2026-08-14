"""Module for training the ML model."""
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from src.constants import Target
from src.data_ingestion import DataIngestor
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


class ModelTrainer:
    """Class responsible for training the ML model."""

    def __init__(self, data_path: str, model_path: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.ingestor = DataIngestor(self.data_path)
        self.engineer = FeatureEngineer()

    def load_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads, cleans, and splits the dataset."""
        logger.info("Starting data loading and preprocessing pipeline.")
        try:
            raw_data = self.ingestor.extract()
            cleaned_data = self.engineer.clean(raw_data)
            prepared_data = self.engineer.transform(cleaned_data, fit=True)

            x_features = prepared_data[self.engineer.FEATURES]
            y = prepared_data[Target.DEFAULT]

            x_train, x_test, y_train, y_test = train_test_split(
                x_features, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info("Data loading and splitting completed successfully.")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logger.error("Data loading failed: %s", e)
            raise

    def train_model(self) -> Dict[str, float]:
        """Trains the model and saves the artifact."""
        logger.info("Starting model training process.")
        try:
            x_train, x_test, y_train, y_test = self.load_data()

            model = LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            )
            model.fit(x_train, y_train)
            logger.info("Model fitting completed.")

            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "calibration_brier": float(brier_score_loss(y_test, y_prob)),
            }

            logger.info("Model evaluation metrics: %s", metrics)

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            artifact = {
                "model": model,
                "scaler": self.engineer.scaler,
                "metrics": metrics,
            }
            joblib.dump(artifact, self.model_path)
            logger.info("Model artifact saved to %s", self.model_path)

            return metrics
        except Exception as e:
            logger.error("Model training failed: %s", e)
            raise
