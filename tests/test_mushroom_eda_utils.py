# tests/test_mushroom_eda_utils.py

import numpy as np
import pandas as pd
import pytest

from src.mushroom_eda_utils import (
    build_feature_matrices,
    summarize_features,
    drop_veil_type_if_present,
    get_poison_rate_by,
    cramers_v,
    compute_poison_variance_rank,
)


def test_build_feature_matrices_splits_target_correctly():
    train_df = pd.DataFrame(
        {
            "feature_a": ["x", "y", "x"],
            "is_poisonous": [0, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "feature_a": ["y", "y"],
            "is_poisonous": [1, 0],
        }
    )

    X_train, X_test, y_train, y_test = build_feature_matrices(train_df, test_df)

    # Target column removed from X, present in y
    assert "is_poisonous" not in X_train.columns
    assert "is_poisonous" not in X_test.columns
    assert y_train.tolist() == [0, 1, 0]
    assert y_test.tolist() == [1, 0]


def test_summarize_features_counts_categories_and_modes():
    X_train = pd.DataFrame(
        {
            "odor": ["a", "a", "b", "b", "b"],
            "gill_size": ["n", "n", "b", "n", "n"],
        }
    )

    summary = summarize_features(X_train)

    # Expected features present
    assert set(summary["feature"]) == {"odor", "gill_size"}

    odor_row = summary[summary["feature"] == "odor"].iloc[0]
    gill_row = summary[summary["feature"] == "gill_size"].iloc[0]

    # odor has 2 categories: a, b; most frequent is b with count 3
    assert odor_row["n_categories"] == 2
    assert odor_row["most_frequent_category"] == "b"
    assert odor_row["most_frequent_count"] == 3

    # gill_size has 2 categories: n, b; most frequent is n with count 4
    assert gill_row["n_categories"] == 2
    assert gill_row["most_frequent_category"] == "n"
    assert gill_row["most_frequent_count"] == 4


def test_drop_veil_type_if_present_removes_column_safely():
    train_df = pd.DataFrame(
        {
            "cap_shape": ["x", "b"],
            "veil_type": ["p", "p"],
            "is_poisonous": [0, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "cap_shape": ["x"],
            "veil_type": ["p"],
            "is_poisonous": [1],
        }
    )

    new_train, new_test = drop_veil_type_if_present(train_df, test_df)

    assert "veil_type" not in new_train.columns
    assert "veil_type" not in new_test.columns
    # Original dataframes not mutated
    assert "veil_type" in train_df.columns
    assert "veil_type" in test_df.columns


def test_get_poison_rate_by_computes_correct_fractions():
    df = pd.DataFrame(
        {
            "odor": ["a", "a", "b", "b"],
            "is_poisonous": [0, 1, 1, 1],
        }
    )
    # odor 'a': 1 edible, 1 poisonous -> 0.5 / 0.5
    # odor 'b': 0 edible, 2 poisonous -> 0.0 / 1.0

    rates = get_poison_rate_by(df, "odor")

    # Rows sorted descending by poisonous_frac, so 'b' should be first
    assert list(rates.index) == ["b", "a"]

    b_row = rates.loc["b"]
    a_row = rates.loc["a"]

    assert np.isclose(b_row["edible_frac"], 0.0)
    assert np.isclose(b_row["poisonous_frac"], 1.0)

    assert np.isclose(a_row["edible_frac"], 0.5)
    assert np.isclose(a_row["poisonous_frac"], 0.5)


def test_cramers_v_is_one_for_perfect_association():
    df = pd.DataFrame(
        {
            "feature1": ["x", "x", "y", "y"],
            "feature2": ["x", "x", "y", "y"],
        }
    )

    v = cramers_v(df, "feature1", "feature2")

    assert 0.99 <= v <= 1.01  # allow small numeric wiggle


def test_compute_poison_variance_rank_orders_features_by_variance():
    df = pd.DataFrame(
        {
            "odor": ["a", "a", "b", "b"],
            "veil_type": ["p", "p", "p", "p"],  # constant -> variance 0
            "is_poisonous": [0, 1, 1, 1],
        }
    )

    rank_df = compute_poison_variance_rank(df, target_col="is_poisonous")

    # odor should have higher variance than veil_type
    top_feature = rank_df.iloc[0]["feature"]
    bottom_feature = rank_df.iloc[-1]["feature"]

    assert top_feature == "odor"
    assert bottom_feature == "veil_type"

    # veil_type should have zero poison_variance
    veil_row = rank_df[rank_df["feature"] == "veil_type"].iloc[0]
    assert np.isclose(veil_row["poison_variance"], 0.0)
