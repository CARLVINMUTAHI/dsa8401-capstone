"""
tests/test_pipeline.py
======================
Unit tests for the data pipeline logic (01_pipeline.py).
Tests run on synthetic data — no real KDHS file required.
"""

import sys  # noqa: E401
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Import functions from pipeline
from pipeline_helpers import (  # noqa: E402
    filter_sample,
    engineer_features,
    select_features,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

def make_sample_df(n: int = 200) -> pd.DataFrame:
    """Generate synthetic KDHS-like dataframe for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "v001": rng.integers(1, 300, n),
        "v002": rng.integers(1, 50, n),
        "v005": rng.integers(500000, 3000000, n),
        "v024": rng.integers(1, 48, n),
        "v025": rng.integers(1, 3, n),
        "b4":   rng.integers(1, 3, n),
        "b19":  rng.integers(0, 60, n),
        "bord": rng.integers(1, 10, n),
        "b11":  rng.integers(12, 60, n).astype(float),
        "m18":  rng.integers(1, 6, n).astype(float),
        "m19":  rng.uniform(1.5, 5.0, n),
        "h11":  rng.integers(0, 2, n),
        "h22":  rng.integers(0, 2, n),
        "hw70": rng.integers(-500, 300, n),   # HAZ x100
        "hw71": rng.integers(-400, 300, n),   # WAZ x100
        "hw72": rng.integers(-400, 300, n),   # WHZ x100
        "hw13": np.zeros(n, dtype=int),        # all valid measurements
        "v106": rng.integers(0, 4, n),
        "v149": rng.integers(0, 4, n),
        "v437": rng.uniform(400, 900, n),      # weight x10
        "v438": rng.uniform(1400, 1800, n),    # height x10
        "v404": rng.integers(0, 2, n),
        "v113": rng.integers(10, 72, n),
        "v116": rng.integers(10, 44, n),
        "v119": rng.integers(0, 2, n),
        "v137": rng.integers(0, 8, n),
        "v190": rng.integers(1, 6, n),
        "v191": rng.uniform(0, 1, n),
        "v151": rng.integers(1, 3, n),
        "v152": rng.integers(20, 80, n).astype(float),
    })


# ── TESTS: filter_sample ──────────────────────────────────────────────────────

class TestFilterSample:
    def test_removes_invalid_hw13(self):
        df = make_sample_df(100)
        df.loc[:20, "hw13"] = 1  # mark 21 as invalid
        result = filter_sample(df)
        assert (result["hw13"] == 0).all()

    def test_removes_implausible_haz(self):
        df = make_sample_df(100)
        df.loc[0, "hw70"] = -700   # below HAZ_MIN
        df.loc[1, "hw70"] = 700    # above HAZ_MAX
        result = filter_sample(df)
        assert len(result) <= 98

    def test_removes_children_over_59_months(self):
        df = make_sample_df(100)
        df.loc[:5, "b19"] = 60     # 6 children exactly 60 months
        result = filter_sample(df)
        assert (result["b19"] < 60).all()

    def test_returns_dataframe(self):
        df = make_sample_df(100)
        result = filter_sample(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_rows_lost_when_all_valid(self):
        df = make_sample_df(50)
        df["hw13"] = 0
        df["hw70"] = 0
        df["hw71"] = 0
        df["hw72"] = 0
        df["b19"] = 24
        result = filter_sample(df)
        assert len(result) == 50


# ── TESTS: engineer_features ──────────────────────────────────────────────────

class TestEngineerFeatures:
    @pytest.fixture
    def filtered_df(self):
        df = make_sample_df(200)
        return filter_sample(df)

    def test_outcome_columns_created(self, filtered_df):
        df = engineer_features(filtered_df)
        assert "stunted" in df.columns
        assert "wasted" in df.columns
        assert "underweight" in df.columns

    def test_stunted_binary(self, filtered_df):
        df = engineer_features(filtered_df)
        assert set(df["stunted"].unique()).issubset({0, 1})

    def test_wash_score_range(self, filtered_df):
        df = engineer_features(filtered_df)
        assert df["wash_score"].between(0, 3).all()

    def test_maternal_bmi_clipped(self, filtered_df):
        df = engineer_features(filtered_df)
        assert df["maternal_bmi"].between(10, 50).all()

    def test_no_nulls_in_key_features(self, filtered_df):
        df = engineer_features(filtered_df)
        key_cols = ["stunted", "wasted", "underweight",
                    "child_age_months", "wealth_index", "maternal_edu"]
        assert df[key_cols].isnull().sum().sum() == 0

    def test_county_column_integer(self, filtered_df):
        df = engineer_features(filtered_df)
        assert df["county"].dtype in [np.int32, np.int64, int]

    def test_survey_weight_positive(self, filtered_df):
        df = engineer_features(filtered_df)
        assert (df["survey_weight"] > 0).all()

    def test_stunting_threshold(self, filtered_df):
        """Children with HAZ x100 < -200 must be flagged stunted."""
        df = filtered_df.copy()
        df["hw70"] = -300  # all below threshold
        df = engineer_features(df)
        assert df["stunted"].all()

    def test_non_stunting_threshold(self, filtered_df):
        """Children with HAZ x100 >= -200 must not be flagged stunted."""
        df = filtered_df.copy()
        df["hw70"] = 0  # all above threshold
        df = engineer_features(df)
        assert not df["stunted"].any()


# ── TESTS: select_features ────────────────────────────────────────────────────

class TestSelectFeatures:
    @pytest.fixture
    def engineered_df(self):
        df = make_sample_df(200)
        df = filter_sample(df)
        return engineer_features(df)

    def test_returns_tuple(self, engineered_df):
        result = select_features(engineered_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_feature_list_non_empty(self, engineered_df):
        feat_cols, _ = select_features(engineered_df)
        assert len(feat_cols) > 0

    def test_county_in_features(self, engineered_df):
        feat_cols, _ = select_features(engineered_df)
        assert "county" in feat_cols

    def test_outcome_cols_in_df(self, engineered_df):
        _, df_model = select_features(engineered_df)
        for col in ["stunted", "wasted", "underweight"]:
            assert col in df_model.columns

    def test_no_nulls_in_features(self, engineered_df):
        feat_cols, df_model = select_features(engineered_df)
        assert df_model[feat_cols].isnull().sum().sum() == 0
