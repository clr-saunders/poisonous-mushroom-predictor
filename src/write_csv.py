import os
import pandas as pd

# elements of the code below are attributed to Tiffany Timbers, DSCI 522, Milestone 4 Example. 

def write_csv(df: pd.DataFrame, directory: str, file: str, index: bool = False):
    """
    Save a Pandas DataFrame to a CSV file in the specified directory

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to save to a csv
    directory : str
        The directory folder to save the csv file to
    file : str
        The name of the saved csv file
    index : bool, optional
        Whether to include the index of the df in the csv file. Default is false. 

    Raises
    ------
    TypeError
        If the input df is not a Pandas DataFrame or is empty 
    TypeError
        If the input directory or file are not a string
    TypeError
        If the input index is not a bool
    ValueError
        If the file name does not include .csv at the end
    FileNotFoundError
        If the input directory does not exist
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be type pandas DataFrame.") 
    if not isinstance(directory, str):
        raise TypeError("Input directory must be a string")
    if not isinstance(file, str):
        raise TypeError("Input file must be a string")
    if not isinstance(index, bool):
        raise TypeError("Input index must be type bool")
    if not file.endswith(".csv"):
        raise ValueError("Input file must be a string that ends with '.csv'")
    if df.empty:
        raise ValueError("Input df must not be empty")
    if not os.path.exists(directory):
        raise FileNotFoundError("Input directory does not exist")

    filepath = os.path.join(directory, file)
    df.to_csv(filepath, index=index)