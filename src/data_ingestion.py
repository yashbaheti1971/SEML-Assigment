"""Data ingestion module for reading input CSVs."""
import logging
from pathlib import Path

import pandas as pd

from src.constants import Target

from src.logging_config import get_logger

logger = get_logger(__name__)



class DataIngestor:
    """Class responsible for reading data and performing basic validations."""

    EXPECTED_COLUMNS = [
        "applicant_id",
        "credit_score",
        "annual_income",
        "loan_amount",
        "dti_ratio",
        "employment_tenure",
        Target.DEFAULT,
    ]

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def extract(self) -> pd.DataFrame:
        """Reads CSV data and returns a pandas DataFrame."""
        logger.info("Starting data extraction from %s", self.csv_path)
        if not self.csv_path.exists():
            logger.error("Data file not found: %s", self.csv_path)
            raise FileNotFoundError(f"Data CSV not found at {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
            logger.info("Successfully loaded %d records from %s", len(df), self.csv_path)

            # Basic Data Quality metric: schema validation
            missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
            if missing_cols:
                logger.error(
                    "Schema validation failed. Missing columns: %s", missing_cols
                )
                raise ValueError(
                    f"Schema validation failed. Missing columns: {missing_cols}"
                )

            logger.info("Schema validation passed.")
            return df
        except Exception as e:
            logger.error("Failed to read CSV data: %s", e)
            raise
