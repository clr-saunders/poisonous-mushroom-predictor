"""
split_and_eda.py

Reproducible EDA for the mushroom classification project:

- EDA on the training set (head, info, shape)
- Check class balance and target distribution
- Feature summary table
- Drop non-informative 'veil_type' feature from X (if present)
- Save updated mushroom_train/mushroom_test with any changes
- Save X_train, X_test, y_train, y_test to data/processed/ (optional, remove if not needed)
- Compute Cramer's V correlation matrix and save a heatmap
- Generate stacked bar charts for selected categorical features

This script assumes pre-split data are available as:
    data/processed/mushroom_train.csv
    data/processed/mushroom_test.csv
with a binary target column `is_poisonous` (0 = edible, 1 = poisonous).
"""

from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import altair as alt

from src.mushroom_eda_utils import (
    build_feature_matrices,
    summarize_features,
    drop_veil_type_if_present,
    get_poison_rate_by,
    cramers_v,
    compute_poison_variance_rank,
)


# -------------------------------------------------------------------
# Configuration for reproducibility and paths
# -------------------------------------------------------------------

RANDOM_STATE = 123
EXPECTED_POISONOUS = 0.482   # from UCI description
EXPECTED_TOL = 0.02          # allowed absolute deviation in poisonous proportion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TRAIN_CSV = PROCESSED_DIR / "mushroom_train.csv"
INPUT_TEST_CSV = PROCESSED_DIR / "mushroom_test.csv"

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def load_train_test_data(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train and test mushroom datasets."""
    if not train_path.exists():
        raise FileNotFoundError(f"Expected train data at {train_path}.")
    if not test_path.exists():
        raise FileNotFoundError(f"Expected test data at {test_path}.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_eda_train(train_df: pd.DataFrame) -> None:
    """Print basic EDA diagnostics on the training set."""
    print("\n=== EDA on training data ===")
    print("\nDataFrame info():")
    print(train_df.info())
    print("\nData shape (rows, columns):", train_df.shape)
    print("\nFirst five rows:")
    print(train_df.head())


def eda_after_split(train_df: pd.DataFrame) -> None:
    """Check class balance and validate target distribution in the training set."""
    print("\n=== EDA after data splitting (training set) ===")
    counts = train_df["is_poisonous"].value_counts()
    props = train_df["is_poisonous"].value_counts(normalize=True)

    counts.index = counts.index.map({0: "edible (0)", 1: "poisonous (1)"})
    props.index = props.index.map({0: "edible (0)", 1: "poisonous (1)"})

    print("\nTraining set class counts:")
    print(counts)

    print("\nTraining set class proportions:")
    print(props.round(3))

    train_poisonous = train_df["is_poisonous"].mean()
    diff = abs(train_poisonous - EXPECTED_POISONOUS)
    assert diff <= EXPECTED_TOL, (
        f"Training poisonous proportion ({train_poisonous:.3f}) differs from "
        f"expected ({EXPECTED_POISONOUS:.3f}) by more than {EXPECTED_TOL:.2f}."
    )
    print(
        f"\n[OK] Training poisonous proportion ({train_poisonous:.3f}) "
        f"is close to expected ({EXPECTED_POISONOUS:.3f})."
    )


# def build_feature_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame):
#     """Construct X/y matrices from train/test splits."""
#     X_train = train_df.drop(columns=["is_poisonous"])
#     y_train = train_df["is_poisonous"]

#     X_test = test_df.drop(columns=["is_poisonous"])
#     y_test = test_df["is_poisonous"]

#     return X_train, X_test, y_train, y_test


# def summarize_features(X_train: pd.DataFrame) -> pd.DataFrame:
#     """
#     Summarize categorical feature characteristics:
#     - number of categories
#     - most frequent category
#     - its count
#     """
#     summary_rows: list[dict] = []
#     for col in X_train.columns:
#         vc = X_train[col].value_counts()
#         summary_rows.append(
#             {
#                 "feature": col,
#                 "n_categories": vc.shape[0],
#                 "most_frequent_category": vc.index[0],
#                 "most_frequent_count": int(vc.iloc[0]),
#             }
#         )

#     feature_summary = (
#         pd.DataFrame(summary_rows)
#         .sort_values("n_categories", ascending=False)
#         .reset_index(drop=True)
#     )
#     feature_summary.index = feature_summary.index + 1
#     return feature_summary


# def drop_veil_type_if_present(
#     train_df: pd.DataFrame, test_df: pd.DataFrame
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Drop 'veil_type' if present (single-category, non-informative feature)
#     from both train and test dataframes.
#     """
#     if "veil_type" in train_df.columns:
#         train_df = train_df.drop(columns=["veil_type"])
#         if "veil_type" in test_df.columns:
#             test_df = test_df.drop(columns=["veil_type"])
#         print("\nDropped 'veil_type' column from train and test data (single level).")
#     return train_df, test_df


def save_updated_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Overwrite mushroom_train/test with any updated columns (e.g., veil_type dropped)."""
    train_df.to_csv(PROCESSED_DIR / "mushroom_train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "mushroom_test.csv", index=False)
    print("\nUpdated mushroom_train.csv and mushroom_test.csv in data/processed/.")


# def get_poison_rate_by(train_df: pd.DataFrame, feature: str) -> pd.DataFrame:
#     """
#     Compute the proportion of poisonous and edible mushrooms for each category
#     of a given feature, based on the training set.
#     """
#     category_poisonous = pd.crosstab(
#         train_df[feature],
#         train_df["is_poisonous"],
#         normalize="index",
#     )
#     # 0 = edible, 1 = poisonous
#     category_poisonous.columns = ["edible_frac", "poisonous_frac"]
#     return category_poisonous.sort_values("poisonous_frac", ascending=False)


def stacked_poison_chart(
    train_df: pd.DataFrame,
    feature: str,
    feature_label: str | None = None,
    category_labels: dict[str, str] | None = None,
):
    """
    Create a stacked bar chart (Altair) showing edible vs poisonous fractions
    for each category of a given feature.
    """
    if feature_label is None:
        feature_label = feature.replace("_", " ").title()

    ct = get_poison_rate_by(train_df, feature).reset_index()

    if category_labels is not None:
        display_col = f"{feature}_label"
        ct[display_col] = ct[feature].map(category_labels).fillna(ct[feature])
    else:
        display_col = feature

    long_df = ct.melt(
        id_vars=display_col,
        value_vars=["edible_frac", "poisonous_frac"],
        var_name="class",
        value_name="fraction",
    )

    long_df["class_label"] = long_df["class"].map(
        {"edible_frac": "Edible", "poisonous_frac": "Poisonous"}
    )

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{display_col}:N", title=feature_label, sort="-x"),
            x=alt.X("fraction:Q", title="Fraction", axis=alt.Axis(format=".2f")),
            color=alt.Color(
                "class_label:N",
                title="Mushroom class",
                scale=alt.Scale(
                    domain=["Edible", "Poisonous"],
                    range=["#17BECF", "#7E1E9C"],
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{display_col}:N", title=feature_label),
                alt.Tooltip("class_label:N", title="Class"),
                alt.Tooltip("fraction:Q", title="Fraction", format=".2f"),
            ],
        )
        .properties(width=450)
    )
    return chart


def save_stacked_charts(train_df: pd.DataFrame) -> None:
    """Generate and save stacked bar charts for selected features as PNG files."""
    odor_map = {
        "a": "almond",
        "c": "creosote",
        "f": "foul",
        "l": "anise",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy",
        "y": "fishy",
    }

    habitat_map = {
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
        "d": "woods",
    }

    gill_size_map = {
        "b": "broad",
        "n": "narrow",
    }

    bruises_map = {
        "t": "bruises",
        "f": "no bruises",
    }

    population_map = {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary",
    }

    charts = {
        "odor": (odor_map, "Odor Category", "stacked_odor.png"),
        "gill_size": (gill_size_map, "Gill Size", "stacked_gill_size.png"),
        "habitat": (habitat_map, "Habitat", "stacked_habitat.png"),
        "bruises": (bruises_map, "Bruises", "stacked_bruises.png"),
        "population": (population_map, "Population", "stacked_population.png"),
    }

    for feature, (mapping, label, filename) in charts.items():
        chart = stacked_poison_chart(
            train_df=train_df,
            feature=feature,
            feature_label=label,
            category_labels=mapping,
        )
        out_path = FIGURES_DIR / filename
        chart.save(out_path)  # extension `.png` tells Altair to export PNG
        print(f"Saved stacked chart for '{feature}' to {out_path}")

# def compute_and_save_poison_variance_rank(train_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Rank features by how strongly they separate poisonous vs edible mushrooms.

#     Uses the variance of the 'poisonous_frac' across categories of each feature
#     as an association measure.
#     """
#     feature_cols = [c for c in train_df.columns if c != "is_poisonous"]

#     feature_scores: list[dict] = []

#     for col in feature_cols:
#         ct = get_poison_rate_by(train_df, col)
#         score = ct["poisonous_frac"].var()
#         feature_scores.append(
#             {
#                 "feature": col,
#                 "poison_variance": round(float(score), 2),
#             }
#         )

#     feature_importance_eda = (
#         pd.DataFrame(feature_scores)
#         .sort_values("poison_variance", ascending=False)
#         .reset_index(drop=True)
#     )
#     feature_importance_eda.index = feature_importance_eda.index + 1

#     out_path = TABLES_DIR / "feature_poison_variance_rank.csv"
#     feature_importance_eda.to_csv(out_path, index=True)
#     print(f"Saved poison-variance feature ranking to {out_path}")

#     return feature_importance_eda

def compute_and_save_poison_variance_rank(
    train_df: pd.DataFrame,
    out_path: Path | None = None,
    target_col: str = "is_poisonous",
) -> pd.DataFrame:
    """
    Compute and persist a ranking of features by their poison-variance score.

    This is a thin I/O wrapper around ``compute_poison_variance_rank`` from
    ``mushroom_eda_utils``. It computes the ranking from the training data and
    writes it to CSV for downstream reporting.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing feature columns and the target column.
    out_path : pathlib.Path or None, default=None
        Output CSV path. If None, defaults to:
        ``TABLES_DIR / "feature_poison_variance_rank.csv"``.
    target_col : str, default="is_poisonous"
        Name of the target column used to compute class proportions.

    Returns
    -------
    pandas.DataFrame
        Ranked features with columns ``['feature', 'poison_variance']``.
    """
    if out_path is None:
        out_path = TABLES_DIR / "feature_poison_variance_rank.csv"

    feature_importance_eda = compute_poison_variance_rank(
        train_df=train_df,
        target_col=target_col,
    )

    # Save with the rank index (1-based) included as the first column
    feature_importance_eda.to_csv(out_path, index=True)
    print(f"Saved poison-variance feature ranking to {out_path}")

    return feature_importance_eda



# def cramers_v(df: pd.DataFrame, feature1: str, feature2: str) -> float:
#     """Compute Cramér's V between two categorical features."""
#     table = pd.crosstab(df[feature1], df[feature2])
#     chi2, _, _, _ = chi2_contingency(table)
#     n = table.sum().sum()
#     phi2 = chi2 / n
#     r, k = table.shape
#     return float(np.sqrt(phi2 / min(k - 1, r - 1)))


# def compute_and_save_cramers_matrix(train_df: pd.DataFrame) -> None:
#     """
#     Compute Cramér's V matrix for all pairwise feature/target combinations
#     and save both the matrix and a PNG heatmap.
#     """
#     cols = list(train_df.columns)
#     feature_cols = cols  # includes target; that's fine for inspection
#     pairs = list(itertools.combinations(feature_cols, 2))

#     cramers_matrix = pd.DataFrame(
#         np.eye(len(feature_cols)),
#         columns=feature_cols,
#         index=feature_cols,
#     )

#     for f1, f2 in pairs:
#         v = cramers_v(train_df, f1, f2)
#         cramers_matrix.loc[f1, f2] = v
#         cramers_matrix.loc[f2, f1] = v

#     # Save raw matrix (table)
#     matrix_path = TABLES_DIR / "cramers_v_matrix.csv"
#     cramers_matrix.to_csv(matrix_path)
#     print(f"\nSaved Cramér's V matrix to {matrix_path}")

#     # Heatmap PNG
#     cramers_long = cramers_matrix.reset_index().melt(id_vars="index")
#     cramers_long.columns = ["feature_1", "feature_2", "cramers_v"]

#     heatmap = (
#         alt.Chart(cramers_long)
#         .mark_rect()
#         .encode(
#             x=alt.X("feature_1:N", sort=feature_cols, title="Feature 1"),
#             y=alt.Y("feature_2:N", sort=feature_cols, title="Feature 2"),
#             color=alt.Color(
#                 "cramers_v:Q",
#                 scale=alt.Scale(domain=[0, 1], scheme="purpleorange"),
#                 title="Cramér's V",
#             ),
#         )
#         .properties(width=300, height=300)
#     )

#     heatmap_path = FIGURES_DIR / "cramers_v_heatmap.png"
#     heatmap.save(heatmap_path)
#     print(f"Saved Cramér's V heatmap to {heatmap_path}")
    

def compute_and_save_cramers_matrix(train_df: pd.DataFrame) -> None:
    """
    Compute Cramer's V matrix for all pairwise feature/target combinations
    and save only the PNG heatmap.
    """
    cols = list(train_df.columns)
    feature_cols = cols
    pairs = list(itertools.combinations(feature_cols, 2))

    cramers_matrix = pd.DataFrame(
        np.eye(len(feature_cols)),
        columns=feature_cols,
        index=feature_cols,
    )

    for f1, f2 in pairs:
        v = cramers_v(train_df, f1, f2)
        cramers_matrix.loc[f1, f2] = v
        cramers_matrix.loc[f2, f1] = v

    # Heatmap PNG only
    cramers_long = cramers_matrix.reset_index().melt(id_vars="index")
    cramers_long.columns = ["feature_1", "feature_2", "cramers_v"]

    heatmap = (
        alt.Chart(cramers_long)
        .mark_rect()
        .encode(
            x=alt.X("feature_1:N", sort=feature_cols, title="Feature 1"),
            y=alt.Y("feature_2:N", sort=feature_cols, title="Feature 2"),
            color=alt.Color(
                "cramers_v:Q",
                scale=alt.Scale(domain=[0, 1], scheme="purpleorange"),
                title="Cramer's V",
            ),
        )
        .properties(width=300, height=300)
    )

    heatmap_path = FIGURES_DIR / "cramers_v_heatmap.png"
    heatmap.save(heatmap_path)
    print(f"Saved Cramer's V heatmap to {heatmap_path}")


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------


def main():
    # Load pre-split train and test data
    train_df, test_df = load_train_test_data(INPUT_TRAIN_CSV, INPUT_TEST_CSV)

    # EDA on the training set
    basic_eda_train(train_df)

    # EDA after splitting (class balance + target distribution)
    eda_after_split(train_df)

    # Drop veil_type from both train and test (if present)
    train_df, test_df = drop_veil_type_if_present(train_df, test_df)

    # Save updated train/test (so downstream uses the same cleaned feature set)
    save_updated_train_test(train_df, test_df)

    # Build feature matrices for summary & correlation
    X_train, X_test, y_train, y_test = build_feature_matrices(train_df, test_df)

    # Feature summary (table)
    feature_summary = summarize_features(X_train)
    summary_path = TABLES_DIR / "feature_summary.csv"
    feature_summary.to_csv(summary_path)
    print(f"\nSaved feature summary to {summary_path}")

    # Stacked bar charts for selected features (PNG)
    save_stacked_charts(train_df)
    
    # Feature ranking by poison variance (table)
    compute_and_save_poison_variance_rank(train_df)

    # Cramér's V matrix (table) and heatmap (PNG)
    compute_and_save_cramers_matrix(train_df)


if __name__ == "__main__":
    main()
