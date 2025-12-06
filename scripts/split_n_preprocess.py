import pandas as pd
import pandera.pandas as pa
import numpy as np
import click
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
def main(raw_data):
    """Preprocesses data from the read_data script and splits data """
    
    # Expected data column names
    dataset_col_names = [
        "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
        "gill_attachment", "gill_spacing", "gill_size", "gill_color",
        "stalk_shape", "stalk_root",
        "stalk_surface_above_ring", "stalk_surface_below_ring",
        "stalk_color_above_ring", "stalk_color_below_ring",
        "veil_type", "veil_color",
        "ring_number", "ring_type",
        "spore_print_color", "population", "habitat"
    ]
    
    df = pd.read_csv(raw_data, names=dataset_col_names)
    
    df.head()
    
    # Data Validation Checks
    ''' No duplicate observations '''
    duplicates = df.duplicated().sum()
    assert duplicates == 0, (
        f"Found {duplicates} full-row duplicates."
    )
    
    '''Target follows expected distribution'''
    actual_distribution = (
        df["class"]
        .value_counts(normalize=True)
        .round(3)
        .to_dict()
    )
    
    expected_distribution = {'e': 0.518, 'p':0.482}
    
    for cls, expected_prop in expected_distribution.items():
        observed_prop = actual_distribution.get(cls, 0.0)
        assert np.isclose(observed_prop, expected_prop, atol=0.01), (
            f"Class '{cls}': observed proportion {observed_prop} "
            f"differs from expected {expected_prop}."
        )
    
    '''Correct category levels check (i.e., no string mismatches or single values)'''
    
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
    
        "habitat": ["g", "l", "m", "p", "u", "w", "d"]
    }

    for col, allowed in allowed_values.items():
    
    
        invalid = set(df[col].dropna().unique()) - set(allowed)
    
    
        assert not invalid, f"{col} has invalid values: {invalid}"

    
    # Convert "?" markers to proper NaN in case any other should occur apart from the known `stalk_root`_ just thinking real-world applications
    
    df = df.replace("?", np.nan)
    
    
    # Changing the target to numeric values: poisonous=1, edible=0
    df["is_poisonous"] = df["class"].map({"p": 1, "e": 0})
    
    # Drop rows with NaN in the target column (none expected, but safe)
    df = df.dropna(subset=["is_poisonous"])
    
    
    # Drop stalk_root column due to missing values and original target column "class" because numeric column of target values "is_poisonous" added to replace it
    df.drop(columns=['stalk_root', 'class'], inplace=True)
    
    
    # Validate cleaned column names
    # Given that the dataset we read in for this model did not include headers, we assigned the names and hence encoded the correct column names initially upon read in. Here, I confirm the column names match expectations after dropping "stalk_root" and changing the name of "class" to "is_poisonous"
    expected_cols_cleaned = [
        col for col in dataset_col_names if col not in ["stalk_root", "class"]
    ] + ["is_poisonous"]
    
    assert list(df.columns) == expected_cols_cleaned, \
        "The dataframe does not have the expected column names after cleaning."
    
    # Pandera: validate cleaned dataframe schema
    #   - All feature columns are strings
    #   - Target column is integer and only takes values {0, 1}
    
    schema = pa.DataFrameSchema(
        {
            # All predictor/feature columns: stored as strings
            **{
                col: pa.Column(str)
                for col in expected_cols_cleaned
                if col != "is_poisonous"
            },
            # Target column: must be integer and in the set {0, 1}
            "is_poisonous": pa.Column(
                int,
                pa.Check.isin([0, 1]),
            ),
        }
    )
    
    # Validate the cleaned dataframe against the schema
    schema.validate(df, lazy=True)
    
    # Pandera: validate cleaned dataframe schema    
    # - Ensure there are no completely empty rows
    # - all columns must be present
    # - enforces no missing in `is_poisonous`
    # - enforces <= 5% missingness in any other column
    
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
            # No empty observations: no row where *all* columns are missing
            pa.Check(
                lambda df: ~(df.isna().all(axis=1)).any(),
                element_wise=False,
                error="Empty rows found.",
            ),
            # Missingness not beyond expected threshold (up to max 5% per column)
            pa.Check(
                lambda df: df.isna().mean().max() <= 0.05,
                element_wise=False,
                error="One or more columns exceed 5% missingness.",
            ),
        ],
    )
    
    # Validate the cleaned dataframe (will raise if check fails)
    df = mushroom_schema.validate(df, lazy=True)
    
    
    # Data Splitting 
    
    # Train–test split at the row level
    #   - Keeps all columns together in each split
    #   - Stratify on the target to preserve class balance
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=123,
        stratify=df["is_poisonous"],
    )
    
    # Build X/y for model training and evaluation from splits
    X_train = train_df.drop(columns=["is_poisonous"])
    y_train = train_df["is_poisonous"]
    
    X_test = test_df.drop(columns=["is_poisonous"])
    y_test = test_df["is_poisonous"]
           
if __name__ == '__main__':
    main()
