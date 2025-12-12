# mushroom_eda_utils.py

from __future__ import annotations
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def build_feature_matrices(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split training and test dataframes into feature matrices (X) and
    target vectors (y).

    This function assumes the presence of a binary target column named
    ``'is_poisonous'`` in both input dataframes and returns separate
    feature and target objects suitable for modeling or analysis.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing feature columns and the target column
        ``'is_poisonous'``.

    test_df : pandas.DataFrame
        Test dataset containing feature columns and the target column
        ``'is_poisonous'``.

    Returns
    -------
    X_train : pandas.DataFrame
        Training feature matrix with the target column removed.

    X_test : pandas.DataFrame
        Test feature matrix with the target column removed.

    y_train : pandas.Series
        Target vector for the training data.

    y_test : pandas.Series
        Target vector for the test data.

    Raises
    ------
    KeyError
        If ``'is_poisonous'`` is not present in either input dataframe.
    
    Examples
    --------
    >>> train_df = pd.DataFrame({"odor": ["a", "f"], "is_poisonous": [0, 1]})
    >>> test_df = pd.DataFrame({"odor": ["n"], "is_poisonous": [0]})
    >>> X_train, X_test, y_train, y_test = build_feature_matrices(train_df, test_df)
    >>> list(X_train.columns)
    ['odor']
    >>> y_train.tolist()
    [0, 1]
    """
    if "is_poisonous" not in train_df.columns or "is_poisonous" not in test_df.columns:
        raise KeyError("Both train_df and test_df must contain 'is_poisonous' column.")

    X_train = train_df.drop(columns=["is_poisonous"])
    y_train = train_df["is_poisonous"]

    X_test = test_df.drop(columns=["is_poisonous"])
    y_test = test_df["is_poisonous"]

    return X_train, X_test, y_train, y_test


def summarize_features(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table describing categorical feature properties.

    For each feature column, this function computes:
    - the number of unique categories,
    - the most frequent category,
    - the count of the most frequent category.

    This summary is useful for exploratory data analysis and identifying
    low-variance or non-informative features.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Feature matrix containing only predictor variables.

    Returns
    -------
    pandas.DataFrame
        A summary table with columns:
        ``['feature', 'n_categories', 'most_frequent_category',
        'most_frequent_count']``, sorted by decreasing number of categories.
        The index starts at 1 for readability.
    
    Examples
    --------
    >>> X_train = pd.DataFrame({"odor": ["a", "a", "f"], "bruises": ["t", "f", "t"]})
    >>> out = summarize_features(X_train)
    >>> out.loc[1, "feature"]
    'odor'
    >>> out[out["feature"] == "odor"]["most_frequent_category"].iloc[0]
    'a'
    """
    summary_rows: list[Dict[str, object]] = []

    for col in X_train.columns:
        vc = X_train[col].value_counts()
        if vc.empty:
            # Edge case: all NaN or empty column
            summary_rows.append(
                {
                    "feature": col,
                    "n_categories": 0,
                    "most_frequent_category": None,
                    "most_frequent_count": 0,
                }
            )
        else:
            summary_rows.append(
                {
                    "feature": col,
                    "n_categories": vc.shape[0],
                    "most_frequent_category": vc.index[0],
                    "most_frequent_count": int(vc.iloc[0]),
                }
            )

    feature_summary = (
        pd.DataFrame(summary_rows)
        .sort_values("n_categories", ascending=False)
        .reset_index(drop=True)
    )
    feature_summary.index = feature_summary.index + 1
    return feature_summary


def drop_veil_type_if_present(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove the ``'veil_type'`` feature from both training and test datasets
    if it is present.

    The ``'veil_type'`` feature is known to have only a single category
    in the mushroom dataset and therefore provides no predictive value.
    This function returns copies of the input dataframes and does not
    mutate them in place.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset.

    test_df : pandas.DataFrame
        Test dataset.

    Returns
    -------
    train_df_clean : pandas.DataFrame
        Training dataset with ``'veil_type'`` removed if present.

    test_df_clean : pandas.DataFrame
        Test dataset with ``'veil_type'`` removed if present.
    
    Examples
    --------
    >>> train_df = pd.DataFrame({"veil_type": ["p", "p"], "odor": ["a", "f"]})
    >>> test_df = pd.DataFrame({"veil_type": ["p"], "odor": ["n"]})
    >>> train2, test2 = drop_veil_type_if_present(train_df, test_df)
    >>> "veil_type" in train2.columns
    False
    >>> "veil_type" in test2.columns
    False
    """
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    if "veil_type" in train_copy.columns:
        train_copy = train_copy.drop(columns=["veil_type"])
    if "veil_type" in test_copy.columns:
        test_copy = test_copy.drop(columns=["veil_type"])

    return train_copy, test_copy


def get_poison_rate_by(
    train_df: pd.DataFrame,
    feature: str,
    target_col: str = "is_poisonous",
) -> pd.DataFrame:
    """
    Compute class proportions (edible vs poisonous) for each category
    of a given feature.

    This function calculates, for each category of the specified feature,
    the fraction of observations that are edible (0) and poisonous (1).
    The output is sorted by decreasing poisonous fraction.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing the feature and target columns.

    feature : str
        Name of the categorical feature to analyze.

    target_col : str, default='is_poisonous'
        Name of the binary target column where
        0 = edible and 1 = poisonous.

    Returns
    -------
    pandas.DataFrame
        A dataframe indexed by feature categories with columns:
        ``['edible_frac', 'poisonous_frac']``, sorted by
        ``'poisonous_frac'`` in descending order.

    Raises
    ------
    KeyError
        If the specified feature or target column is not found in the dataframe.
    
    Examples
    --------
    >>> df = pd.DataFrame({"odor": ["a", "a", "f"], "is_poisonous": [0, 1, 1]})
    >>> rates = get_poison_rate_by(df, "odor")
    >>> float(rates.loc["f", "poisonous_frac"])
    1.0
    >>> float(rates.loc["a", "edible_frac"])
    0.5
    """
    if feature not in train_df.columns:
        raise KeyError(f"{feature!r} not found in dataframe.")
    if target_col not in train_df.columns:
        raise KeyError(f"{target_col!r} not found in dataframe.")

    category_poisonous = pd.crosstab(
        train_df[feature],
        train_df[target_col],
        normalize="index",
    )

    # Expect 0 = edible, 1 = poisonous
    # Handle missing columns gracefully for tiny test data
    edible_col = 0
    poisonous_col = 1
    if edible_col not in category_poisonous.columns:
        category_poisonous[edible_col] = 0.0
    if poisonous_col not in category_poisonous.columns:
        category_poisonous[poisonous_col] = 0.0

    category_poisonous = category_poisonous[[edible_col, poisonous_col]]
    category_poisonous.columns = ["edible_frac", "poisonous_frac"]

    return category_poisonous.sort_values("poisonous_frac", ascending=False)


def cramers_v(df: pd.DataFrame, feature1: str, feature2: str) -> float:
    """
    Compute Cramér's V statistic to measure association between two
    categorical variables.

    Cramér's V is a normalized measure of association based on the
    chi-squared statistic and ranges from 0 (no association) to
    1 (perfect association).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing both categorical features.

    feature1 : str
        Name of the first categorical feature.

    feature2 : str
        Name of the second categorical feature.

    Returns
    -------
    float
        Cramér's V value between 0 and 1.

    Raises
    ------
    KeyError
        If either feature is not present in the dataframe.
    
    Examples
    --------
    >>> df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["m", "m", "n", "n"]})
    >>> round(cramers_v(df, "a", "b"), 3)
    1.0
    """
    if feature1 not in df.columns or feature2 not in df.columns:
        raise KeyError("Both feature1 and feature2 must be columns in df.")

    table = pd.crosstab(df[feature1], df[feature2])
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    return float(np.sqrt(phi2 / min(k - 1, r - 1)))


def compute_poison_variance_rank(
    train_df: pd.DataFrame,
    target_col: str = "is_poisonous",
) -> pd.DataFrame:
    """
    Rank features by their ability to distinguish poisonous from edible
    mushrooms.

    For each feature, this function computes the variance of the
    ``'poisonous_frac'`` across its categories. Higher variance indicates
    stronger separation between edible and poisonous classes.

    This function performs only computation and returns the ranking
    without writing to disk, making it suitable for unit testing.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing feature columns and the target column.

    target_col : str, default='is_poisonous'
        Name of the binary target column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns ``['feature', 'poison_variance']``,
        sorted in descending order of ``'poison_variance'``.
        The index starts at 1 for readability.

    Raises
    ------
    KeyError
        If the target column is not present in the dataframe.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "odor": ["a", "a", "f", "f"],
    ...     "bruises": ["t", "f", "t", "f"],
    ...     "is_poisonous": [0, 0, 1, 1],
    ... })
    >>> rank = compute_poison_variance_rank(df)
    >>> rank.iloc[0]["feature"] in ["odor", "bruises"]
    True
    """
    if target_col not in train_df.columns:
        raise KeyError(f"{target_col!r} not found in dataframe.")

    feature_cols = [c for c in train_df.columns if c != target_col]
    feature_scores: list[Dict[str, object]] = []

    for col in feature_cols:
        ct = get_poison_rate_by(train_df, col, target_col=target_col)
        score = ct["poisonous_frac"].var()
        feature_scores.append(
            {
                "feature": col,
                "poison_variance": float(round(score, 2)),
            }
        )

    feature_importance_eda = (
        pd.DataFrame(feature_scores)
        .sort_values("poison_variance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance_eda.index = feature_importance_eda.index + 1
    return feature_importance_eda
