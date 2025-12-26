import pandas as pd

def load_csv(filename: str, section: str | None = None) -> pd.DataFrame:
    """
    Load a custom CSV file with optional section blocks and comment markers.

    This function reads a CSV-like text file that may contain:
    - `[section]` headers to structure the file into named blocks.
    - Comment lines beginning with '//'.
    - Optional header rows introduced by a leading '#', containing comma-separated
      column names.
    - Arbitrary rows of comma-separated values, which are automatically converted
      to floats when possible.

    Notes
    -----
    - Lines consisting only of comments or whitespace are ignored.
    - Rows that contain fewer columns than expected emit a warning but are still
      included in the DataFrame.
    - Values are parsed as floats whenever possible; otherwise, they remain strings.

    Parameters
    ----------
    filename : str
        Path to a CSV file. The filename is validated to ensure it ends with '.csv'.
    section : str | None, optional
        Name of the section to load. If provided, only the data inside the matching
        `[section]` block is parsed. If omitted, the entire file is read.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the parsed data. Column names are taken from the
        header row if present; otherwise, generic integer columns are used.

    Raises
    ------
    ValueError
        If a section is specified but does not exist in the file.
    """
    ...
