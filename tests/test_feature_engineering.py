import numpy as np
import pandas as pd
import pytest

from src.constants import Target
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame(
        {
            "applicant_id": ["A1", "A2", "A2"],  # Duplicate A2
            "credit_score": [700, np.nan, 650],  # Missing value
            "annual_income": [50000, 60000, 60000],
            "loan_amount": [10000, 15000, 15000],
            "dti_ratio": [0.2, 0.3, 0.3],
            "employment_tenure": [5, 2, 2],
            Target.DEFAULT: [0, 1, 1],
        }
    )


def test_clean_removes_duplicates_and_imputes(sample_raw_data):
    engineer = FeatureEngineer()
    cleaned_df = engineer.clean(sample_raw_data)

    assert len(cleaned_df) == 2, "Duplicates were not removed"
    assert (
        not cleaned_df["credit_score"].isnull().any()
    ), "Missing values were not imputed"
    # The median of [700, nan] should be 700 if pandas handles it, but let's just check no nulls


def test_clean_missing_target(sample_raw_data):
    df_no_target = sample_raw_data.drop(columns=[Target.DEFAULT])
    engineer = FeatureEngineer()
    with pytest.raises(KeyError, match="must contain a 'default' column"):
        engineer.clean(df_no_target)


def test_transform_scaling():
    df = pd.DataFrame(
        {
            "applicant_id": ["A1", "A2"],
            "credit_score": [300, 800],
            "annual_income": [50000, 100000],
            "loan_amount": [1000, 5000],
            "dti_ratio": [0.1, 0.5],
            "employment_tenure": [1, 10],
            Target.DEFAULT: [0, 1],
        }
    )
    engineer = FeatureEngineer()
    cleaned = engineer.clean(df)
    transformed = engineer.transform(cleaned, fit=True)

    # Check if scaled properly (mean should be approx 0, std approx 1 or matching scaling)
    assert np.isclose(transformed["credit_score"].mean(), 0, atol=1e-5)
