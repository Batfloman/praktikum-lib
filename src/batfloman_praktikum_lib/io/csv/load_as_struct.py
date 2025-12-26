import pandas as pd

from ..validation import validate_filename
from .load_csv import load_csv

def load_csv_datacluster(filename: str, section: str | None = None):
    from ...structs.dataCluster import DataCluster
    return DataCluster(load_csv(filename, section))

def load_csv_consts(filename, section: str | None = None) -> dict:
    """
        quickly loads sections with a
            '# name, value, error'
        header
        returns: dict with: 'name' -> Measurement(value, error)
    """
    filename = validate_filename(filename, ".csv");

    df = load_csv(filename, section);

    expected_indicies = _expect_consts(df)
    if not "name" in expected_indicies:
        expected_indicies["name"] = None
        print("Warning! No 'name' index found! Using indicies")
        # raise IndexError(f"No 'name' index found! Columns '{df.columns}'")
    if not "value" in expected_indicies:
        raise IndexError(f"No 'value' index found! Columns '{df.columns}'")
    if not "error" in expected_indicies:
        raise IndexError(f"No 'error' index found! Columns '{df.columns}'")

    from ...structs.measurement import Measurement

    consts = {}
    for i, row in df.iterrows():
        name = i if not expected_indicies["name"] else row[expected_indicies["name"]]
        val = row[expected_indicies["value"]]
        err = row[expected_indicies["error"]]
        consts[name] = Measurement(val, err);
    return consts;

# ==================================================
#    helper
# ==================================================


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

