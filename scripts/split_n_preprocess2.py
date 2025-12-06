import click
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pathlib import Path
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.3
RANDOM_STATE = 123


def save_train_test_split_stratified(
    df: pd.DataFrame,
    out_dir: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a reproducible stratified train/test split on the full dataframe
    and save the resulting train/test dataframes to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned mushroom dataframe with `is_poisonous` as the target.
    out_dir : str or Path, optional
        Directory where train and test CSVs should be saved.

    Returns
    -------
    (train_df, test_df)
        DataFrames for the training and test sets.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["is_poisonous"],
    )

    print("\n=== Train/Test split ===")
    print("Train df shape:", train_df.shape)
    print("Test df shape: ", test_df.shape)

    train_df.to_csv(out_dir / "mushroom_train.csv", index=False)
    test_df.to_csv(out_dir / "mushroom_test.csv", index=False)

    return train_df, test_df

def save_column_names_txt(col_names, out_dir):
    """ saves column names in text file """
    col_path = Path(out_dir) / "column_names.txt"
    with open(col_path, "w") as f:
        f.write("\n".join(col_names))


@click.command()
@click.option("--raw-data", type=str, required=True, help="Path to raw data CSV.")
def main(raw_data: str) -> None:
    """Preprocess raw mushroom data, validate it, and save stratified train/test splits."""

    # Expected data column names
    dataset_col_names = [
        "class",
        "cap_shape",
        "cap_surface",
        "cap_color",
        "bruises",
        "odor",
        "gill_attachment",
        "gill_spacing",
        "gill_size",
        "gill_color",
        "stalk_shape",
        "stalk_root",
        "stalk_surface_above_ring",
        "stalk_surface_below_ring",
        "stalk_color_above_ring",
        "stalk_color_below_ring",
        "veil_type",
        "veil_color",
        "ring_number",
        "ring_type",
        "spore_print_color",
        "population",
        "habitat",
    ]

    df = pd.read_csv(raw_data, names=dataset_col_names)

    # ----- Data Validation Checks -----

    # 1. No duplicate observations
    duplicates = df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} full-row duplicates."

    # 2. Target follows expected distribution
    actual_distribution = (
        df["class"].value_counts(normalize=True).round(3).to_dict()
    )
    expected_distribution = {"e": 0.518, "p": 0.482}

    for cls, expected_prop in expected_distribution.items():
        observed_prop = actual_distribution.get(cls, 0.0)
        assert np.isclose(observed_prop, expected_prop, atol=0.01), (
            f"Class '{cls}': observed proportion {observed_prop} "
            f"differs from expected {expected_prop}."
        )

    # 3. Correct category levels (no unexpected strings)
    allowed_values = {
        "class": ["e", "p"],
        "cap_shape": ["b", "c", "x", "f", "k", "s"],
        "cap_surface": ["f", "g", "y", "s"],
        "cap_color": ["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"],
        "bruises": ["t", "f"],
        "odor": ["a", "l", "c", "y", "f", "m", "n", "p", "s"],
        "gill_attachment": ["a", "d", "f", "n"],
        "gill_spacing": ["c", "w", "d"],
        "gill_size": ["b", "n"],
        "gill_color": ["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"],
        "stalk_shape": ["e", "t"],
        "stalk_root": ["b", "c", "u", "e", "z", "r", "?"],  # ? = missing
        "stalk_surface_above_ring": ["f", "y", "k", "s"],
        "stalk_surface_below_ring": ["f", "y", "k", "s"],
        "stalk_color_above_ring": ["n", "b", "c", "g", "o", "p", "e", "w", "y"],
        "stalk_color_below_ring": ["n", "b", "c", "g", "o", "p", "e", "w", "y"],
        "veil_type": ["p", "u"],
        "veil_color": ["n", "o", "w", "y"],
        "ring_number": ["n", "o", "t"],
        "ring_type": ["c", "e", "f", "l", "n", "p", "s", "z"],
        "spore_print_color": ["k", "n", "b", "h", "r", "o", "u", "w", "y"],
        "population": ["a", "c", "n", "s", "v", "y"],
        "habitat": ["g", "l", "m", "p", "u", "w", "d"],
    }

    for col, allowed in allowed_values.items():
        invalid = set(df[col].dropna().unique()) - set(allowed)
        assert not invalid, f"{col} has invalid values: {invalid}"

    # Replace "?" with actual NaN
    df = df.replace("?", np.nan)

    # Encode target: poisonous=1, edible=0
    df["is_poisonous"] = df["class"].map({"p": 1, "e": 0})

    # Drop rows with NaN in the target (defensive)
    df = df.dropna(subset=["is_poisonous"])

    # Drop `stalk_root` and original `class`
    df = df.drop(columns=["stalk_root", "class"])

    # Validate cleaned column names
    expected_cols_cleaned = [
        col for col in dataset_col_names if col not in ["stalk_root", "class"]
    ] + ["is_poisonous"]

    assert list(df.columns) == expected_cols_cleaned, (
        "The dataframe does not have the expected column names after cleaning."
    )

    # ---- Pandera schema: correct types & basic constraints ----
    schema = pa.DataFrameSchema(
        {
            **{
                col: pa.Column(str)
                for col in expected_cols_cleaned
                if col != "is_poisonous"
            },
            "is_poisonous": pa.Column(int, pa.Check.isin([0, 1])),
        }
    )
    schema.validate(df, lazy=True)

    mushroom_schema = pa.DataFrameSchema(
        {
            "cap_shape": pa.Column(str),
            "cap_surface": pa.Column(str),
            "cap_color": pa.Column(str),
            "bruises": pa.Column(str),
            "odor": pa.Column(str),
            "gill_attachment": pa.Column(str),
            "gill_spacing": pa.Column(str),
            "gill_size": pa.Column(str),
            "gill_color": pa.Column(str),
            "stalk_shape": pa.Column(str),
            "stalk_surface_above_ring": pa.Column(str),
            "stalk_surface_below_ring": pa.Column(str),
            "stalk_color_above_ring": pa.Column(str),
            "stalk_color_below_ring": pa.Column(str),
            "veil_type": pa.Column(str),
            "veil_color": pa.Column(str),
            "ring_number": pa.Column(str),
            "ring_type": pa.Column(str),
            "spore_print_color": pa.Column(str),
            "population": pa.Column(str),
            "habitat": pa.Column(str),
            "is_poisonous": pa.Column(
                int,
                pa.Check(
                    lambda s: s.isna().mean() == 0.0,
                    element_wise=False,
                    error="Target 'is_poisonous' contains missing values.",
                ),
                nullable=False,
            ),
        },
        checks=[
            # No empty observations
            pa.Check(
                lambda df_: ~(df_.isna().all(axis=1)).any(),
                element_wise=False,
                error="Empty rows found.",
            ),
            # Missingness not beyond 5% in any column
            pa.Check(
                lambda df_: df_.isna().mean().max() <= 0.05,
                element_wise=False,
                error="One or more columns exceed 5% missingness.",
            ),
        ],
    )

    df = mushroom_schema.validate(df, lazy=True)

    # ------ Stratified train/test split + save ------
    save_train_test_split_stratified(df)

    # ------ Save Dataset Col Names for Use ----
    save_column_names_txt(dataset_col_names, out_dir="data/processed")

if __name__ == "__main__":
    main()
