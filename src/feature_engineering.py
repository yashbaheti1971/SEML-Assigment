"""Feature engineering module for cleaning and transforming data."""
import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.constants import Target

from src.logging_config import get_logger
logger = get_logger(__name__)


class FeatureEngineer:
    """Class responsible for data cleaning and feature transformation."""

    FEATURES = [
        "credit_score",
        "annual_income",
        "loan_amount",
        "dti_ratio",
        "employment_tenure",
    ]

    def __init__(self):
        self.scaler = StandardScaler()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans data by removing duplicates and imputing missing values."""
        logger.info("Starting data cleaning process.")
        out = df.drop_duplicates(subset=["applicant_id"]).copy()

        # Missing value checks and imputation (Data Quality Metric)
        missing_counts = out[self.FEATURES].isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(
                "Found missing values. Imputing with median. Missing counts:\n%s",
                missing_counts[missing_counts > 0]
            )

        out[self.FEATURES] = out[self.FEATURES].fillna(out[self.FEATURES].median())

        if Target.DEFAULT not in out.columns:
            logger.error("Target column 'default' missing.")
            raise KeyError("Input data must contain a 'default' column")

        out[Target.DEFAULT] = out[Target.DEFAULT].astype(int)
        logger.info("Data cleaning completed successfully.")
        return out

    def transform(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scales numeric features using StandardScaler."""
        logger.info("Starting feature transformation (fit=%s).", fit)
        out = df.copy()
        try:
            if fit:
                out[self.FEATURES] = self.scaler.fit_transform(out[self.FEATURES])
                logger.info("Successfully fitted and transformed features.")
            else:
                out[self.FEATURES] = self.scaler.transform(out[self.FEATURES])
                logger.info("Successfully transformed features using existing scaler.")
            return out
        except Exception as e:
            logger.error("Error during feature transformation: %s", e)
            raise
