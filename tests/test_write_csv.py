# As per Skye's direction that we are free to use AI for this task, these tests were written by ChatGPT with some small changes made to naming and some minor adjustments to code made in careful review by Claire Saunders.
# The prompt used was: Write simple and robust tests for the function documented below (I pasted the write_csv function documentation. Use the pytest Python package framework and follow the black style guide for Python. Please ensure you use simple code that is clear and reproducible. Note, these tests will be saved in a tests directory, and will call the function write_csv from the src directory to run the tests. 

import os
import pandas as pd
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv

# Sample Dataframe for Tests:
sample_df = pd.DataFrame({"a" : [1, 2, 3], 
                          "b" : ['cat', 'dog', 'otter'], 
                          "c" : [45.8, 89.0, 2.1]
                         }
                        )
# ----------------------------------------------------------------------
# Success case
# ----------------------------------------------------------------------
def test_write_csv_creates_csv_file(tmp_path):
    directory = tmp_path
    file = "output.csv"

    write_csv(sample_df, str(directory), file)

    # check that file exists
    output_path = directory / file
    assert output_path.exists()

    # Confirm the file content matches
    loaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(loaded, sample_df)


# ----------------------------------------------------------------------
# TypeError: df is not a Pandas DataFrame
# ----------------------------------------------------------------------
def test_write_csv_raises_if_df_not_dataframe(tmp_path):
    directory = str(tmp_path)
    file = "output.csv"

    with pytest.raises(TypeError, match="Input df must be type pandas DataFrame"):
        write_csv("not a df", directory, file)


# ----------------------------------------------------------------------
# ValueError: df is empty
# ----------------------------------------------------------------------
def test_write_csv_raises_if_df_empty(tmp_path):
    df = pd.DataFrame()
    directory = str(tmp_path)
    file = "output.csv"

    with pytest.raises(ValueError, match="Input df must not be empty"):
        write_csv(df, directory, file)


# ----------------------------------------------------------------------
# TypeError: directory not a string
# ----------------------------------------------------------------------
def test_write_csv_raises_if_directory_not_string():

    with pytest.raises(TypeError, match="Input directory must be a string"):
        write_csv(sample_df, 123, "output.csv")


# ----------------------------------------------------------------------
# TypeError: file not a string
# ----------------------------------------------------------------------
def test_write_csv_raises_if_file_not_string(tmp_path):
    directory = str(tmp_path)

    with pytest.raises(TypeError, match="Input file must be a string"):
        write_csv(sample_df, directory, 123)


# ----------------------------------------------------------------------
# ValueError: file does not end with .csv
# ----------------------------------------------------------------------
def test_write_csv_raises_if_file_not_csv(tmp_path):
    directory = str(tmp_path)

    with pytest.raises(ValueError, match="ends with '.csv'"):
        write_csv(sample_df, directory, "wrong.ext")


# ----------------------------------------------------------------------
# FileNotFoundError: directory does not exist
# ----------------------------------------------------------------------
def test_write_csv_raises_if_directory_missing(tmp_path):
    df = pd.DataFrame({"a": [1]})
    missing_dir = tmp_path / "does_not_exist"
    file = "output.csv"

    with pytest.raises(FileNotFoundError):
        write_csv(sample_df, str(missing_dir), file)


# ----------------------------------------------------------------------
# TypeError: index is not bool
# ----------------------------------------------------------------------
def test_write_csv_raises_if_index_not_bool(tmp_path):
    directory = str(tmp_path)
    file = "output.csv"

    with pytest.raises(TypeError):
        write_csv(sample_df, directory, file, index="not a bool")
