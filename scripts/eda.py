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
from pathlib import Path
import sys

# Ensure repo root is on PYTHONPATH so `import src...` works when running scripts directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
    """
    Load pre-split training and test datasets from disk.

    This is a small I/O helper for the EDA pipeline. It validates that both
    input files exist, then loads them with ``pandas.read_csv``.

    Parameters
    ----------
    train_path : pathlib.Path
        Path to the training CSV (e.g., ``data/processed/mushroom_train.csv``).
    test_path : pathlib.Path
        Path to the test CSV (e.g., ``data/processed/mushroom_test.csv``).

    Returns
    -------
    (train_df, test_df) : tuple[pandas.DataFrame, pandas.DataFrame]
        The loaded train and test datasets.

    Raises
    ------
    FileNotFoundError
        If either ``train_path`` or ``test_path`` does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> train_df, test_df = load_train_test_data(
    ...     Path("data/processed/mushroom_train.csv"),
    ...     Path("data/processed/mushroom_test.csv"),
    ... )
    >>> train_df.shape[0] > 0
    True
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Expected train data at {train_path}.")
    if not test_path.exists():
        raise FileNotFoundError(f"Expected test data at {test_path}.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_eda_train(train_df: pd.DataFrame) -> None:
    """
    Print basic exploratory diagnostics for the training dataset.

    Outputs:
    - ``DataFrame.info()``
    - dataset shape
    - first five rows (``head()``)

    This function is intended for human-readable console diagnostics and is
    not designed to return structured values.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset to summarize.

    Returns
    -------
    None
        Prints diagnostics to stdout.

    Examples
    --------
    >>> basic_eda_train(train_df)  # doctest: +SKIP
    === EDA on training data ===
    DataFrame info():
    """
    print("\n=== EDA on training data ===")
    print("\nDataFrame info():")
    print(train_df.info())
    print("\nData shape (rows, columns):", train_df.shape)
    print("\nFirst five rows:")
    print(train_df.head())


def eda_after_split(train_df: pd.DataFrame) -> None:
    """
    Validate training-set class balance and check target distribution tolerance.

    This function is intended to be run after a reproducible train/test split.
    It prints class counts and proportions for the target column
    ``is_poisonous`` and asserts that the observed proportion of poisonous
    samples in the training set is within ``EXPECTED_TOL`` of
    ``EXPECTED_POISONOUS``.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing the target column ``is_poisonous``,
        where 0 = edible and 1 = poisonous.

    Returns
    -------
    None
        Prints diagnostics to stdout.

    Raises
    ------
    KeyError
        If ``is_poisonous`` is missing from ``train_df``.
    AssertionError
        If the training-set poisonous proportion differs from
        ``EXPECTED_POISONOUS`` by more than ``EXPECTED_TOL``.

    Examples
    --------
    >>> eda_after_split(train_df)  # doctest: +SKIP
    === EDA after data splitting (training set) ===
    Training set class counts:
    ...
    [OK] Training poisonous proportion (...) is close to expected (...).
    """
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


def save_updated_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Overwrite the persisted train/test CSVs with updated versions.

    This function is used when the EDA step applies agreed-upon, deterministic
    updates to the pre-split datasets (for example, dropping a non-informative
    feature like ``veil_type``). The updated datasets are written to the
    standard pipeline locations in ``data/processed``.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Updated training dataset to write to ``data/processed/mushroom_train.csv``.
    test_df : pandas.DataFrame
        Updated test dataset to write to ``data/processed/mushroom_test.csv``.

    Returns
    -------
    None
        Side effect only: writes CSV files to disk.

    Side Effects
    ------------
    Writes:
    - ``PROCESSED_DIR / "mushroom_train.csv"``
    - ``PROCESSED_DIR / "mushroom_test.csv"``

    Examples
    --------
    >>> save_updated_train_test(train_df, test_df)  # doctest: +SKIP
    Updated mushroom_train.csv and mushroom_test.csv in data/processed/.
    """
    train_df.to_csv(PROCESSED_DIR / "mushroom_train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "mushroom_test.csv", index=False)
    print("\nUpdated mushroom_train.csv and mushroom_test.csv in data/processed/.")
    

def stacked_poison_chart(
    train_df: pd.DataFrame,
    feature: str,
    feature_label: str | None = None,
    category_labels: dict[str, str] | None = None,
):    
    """
    Create an Altair stacked horizontal bar chart of class composition by category.

    For a given categorical feature, this function computes the within-category
    fractions of edible (0) vs poisonous (1) mushrooms (from ``train_df``) and
    returns a stacked bar chart. Bars are sorted by decreasing poisonous fraction
    to emphasize categories most associated with toxicity.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing the feature column and the binary target
        column ``is_poisonous`` (0 = edible, 1 = poisonous).
    feature : str
        Name of the categorical feature column to visualize (e.g., ``"odor"``).
    feature_label : str, optional
        Human-readable label to use for the y-axis title. If None, a label is
        derived from ``feature`` (underscores replaced with spaces, title-cased).
    category_labels : dict[str, str], optional
        Mapping from raw category codes to display labels (e.g.,
        ``{"a": "almond", "n": "none"}``). If provided, the mapped values are used
        for axis labels; unmapped codes fall back to the raw category value.

    Returns
    -------
    altair.Chart
        An Altair chart object. The caller is responsible for displaying or saving it.

    Notes
    -----
    - Relies on ``get_poison_rate_by(train_df, feature)`` to compute fractions.
    - Assumes the target column is named ``is_poisonous`` and encoded as {0, 1}.

    Examples
    --------
    >>> odor_map = {"a": "almond", "f": "foul", "n": "none"}
    >>> chart = stacked_poison_chart(train_df, "odor", "Odor Category", odor_map)
    >>> chart  # display in notebook/report
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
    """
    Generate and save stacked class-composition charts for selected features.

    This function is a thin orchestration layer around ``stacked_poison_chart``:
    it defines feature-specific label maps, generates one chart per selected
    feature using the training data, and saves each chart as a PNG file in
    ``FIGURES_DIR``.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing the selected feature columns and the binary
        target column ``is_poisonous`` (0 = edible, 1 = poisonous).

    Returns
    -------
    None
        Side effect only: writes PNG files to disk.

    Side Effects
    ------------
    Writes the following files (unless you change the filenames in ``charts``):
    - ``FIGURES_DIR/stacked_odor.png``
    - ``FIGURES_DIR/stacked_gill_size.png``
    - ``FIGURES_DIR/stacked_habitat.png``
    - ``FIGURES_DIR/stacked_bruises.png``
    - ``FIGURES_DIR/stacked_population.png``

    Notes
    -----
    - Requires that ``FIGURES_DIR`` exists (or that the caller created it).
    - PNG export depends on the Altair/Vega export configuration available in
      the runtime environment.

    Examples
    --------
    >>> save_stacked_charts(train_df)
    """
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


def compute_and_save_cramers_matrix(train_df: pd.DataFrame) -> None:
    """
    Compute pairwise Cramér's V associations and save a heatmap as a PNG.

    This function calculates Cramér's V (a chi-squared–based association measure
    for categorical variables) for all pairwise combinations of columns in
    ``train_df`` (including the target column, if present). It then reshapes the
    resulting square matrix to long format and renders an Altair heatmap,
    which is saved to ``FIGURES_DIR/cramers_v_heatmap.png``.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing categorical feature columns and typically
        the binary target column ``is_poisonous``. All columns are treated as
        categorical for association computation.

    Returns
    -------
    None
        Side effect only: writes a PNG heatmap to disk.

    Side Effects
    ------------
    Writes:
    - ``FIGURES_DIR/cramers_v_heatmap.png``

    Notes
    -----
    - Uses ``cramers_v(train_df, f1, f2)`` to compute each pairwise association.
    - The diagonal of the matrix is set to 1.0 by construction.
    - PNG export depends on the Altair/Vega export configuration available in
      the runtime environment (e.g., appropriate renderer/export backend).

    Examples
    --------
    >>> compute_and_save_cramers_matrix(train_df)
    >>> # outputs: results/figures/cramers_v_heatmap.png (assuming FIGURES_DIR is configured)
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
