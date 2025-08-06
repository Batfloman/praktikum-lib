import numpy as np
import pandas as pd

from .validation import ensure_extension, validate_filename

def load_csv(filename: str, section: str = None) -> pd.DataFrame:
    filename = validate_filename(filename, ".csv")

    data = []
    headers = None
    in_section = section is None  # Read everything if no section is specified

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith("[") and line.endswith("]") and section is not None:
                in_section = (line[1:-1] == section)
                continue

            if in_section and line:
                if line.startswith("#"):  
                    temp = line[1:].split(",") # remove '#' and split
                    headers = [x.strip() for x in temp]
                else:
                    temp = line.split(",")
                    data.append([x.strip() for x in temp])

    return pd.DataFrame(data, columns=headers if headers else None)

def load_csv_datacluster(filename: str, section: str = None):
    from ..structs import DataCluster
    return DataCluster(load_csv(filename, section))

def load_csv_consts(filename, section: str = None) -> dict:
    """
        quickly loads sections with a
            '# name, value, error'
        header
        returns: dict with: 'name' -> Measurement(value, error)
    """
    filename = ensure_extension(filename, ".csv");

    df = load_csv(filename, section);

    expected_indicies = _expect_consts(df)
    if not "name" in expected_indicies:
        expected_indicies["name"] = None
        print("Warning! No 'name' index found! Useing indicies")
        # raise IndexError(f"No 'name' index found! Columns '{df.columns}'")
    if not "value" in expected_indicies:
        raise IndexError(f"No 'value' index found! Columns '{df.columns}'")
    if not "error" in expected_indicies:
        raise IndexError(f"No 'error' index found! Columns '{df.columns}'")

    from ..structs import Measurement

    consts = {}
    for i, row in df.iterrows():
        name = i if not expected_indicies["name"] else row[expected_indicies["name"]]
        val = row[expected_indicies["value"]]
        err = row[expected_indicies["error"]]
        consts[name] = Measurement(val, err);
    return consts;

def _expect_consts(df: pd.DataFrame):
    # checks whether a df has 'good enough' indicies for consts method
    expected = {
        "name": ["name", "names", "naems"], 
        "value": ["value", "values", "val", "vals", "vaues"], # Allow abbreviations
        "error": ["error", "errors", "err", "errs", "erorrs"] # Allow abbreviations
    }
    # Normalize column names (strip spaces)
    columns = {col.strip(): col for col in df.columns}

    # Dictionary to store found column names
    found_columns = {}

    # Find matches and store the actual column name used in DataFrame
    for key, variants in expected.items():
        for variant in variants:
            stripped_variant = variant.strip()
            if stripped_variant in columns:
                found_columns[key] = columns[stripped_variant]
                break  # Stop at the first match
    return found_columns;
