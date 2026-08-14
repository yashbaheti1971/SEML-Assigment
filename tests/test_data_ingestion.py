import pandas as pd
import pytest

from src.constants import Target
from src.data_ingestion import DataIngestor


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "applicant_id": ["A1", "A2"],
            "credit_score": [700, 650],
            "annual_income": [50000, 60000],
            "loan_amount": [10000, 15000],
            "dti_ratio": [0.2, 0.3],
            "employment_tenure": [5, 2],
            Target.DEFAULT: [0, 1],
        }
    )
    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def invalid_csv(tmp_path):
    df = pd.DataFrame(
        {
            "applicant_id": ["A1"],
            "credit_score": [700],
            # Missing other required columns
        }
    )
    filepath = tmp_path / "invalid_data.csv"
    df.to_csv(filepath, index=False)
    return filepath


def test_extract_success(sample_csv):
    ingestor = DataIngestor(sample_csv)
    df = ingestor.extract()
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == DataIngestor.EXPECTED_COLUMNS


def test_extract_file_not_found():
    ingestor = DataIngestor("non_existent_file.csv")
    with pytest.raises(FileNotFoundError):
        ingestor.extract()


def test_extract_schema_validation_failure(invalid_csv):
    ingestor = DataIngestor(invalid_csv)
    with pytest.raises(ValueError, match="Schema validation failed"):
        ingestor.extract()
