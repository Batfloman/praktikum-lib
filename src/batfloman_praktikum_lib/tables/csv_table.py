import pandas as pd
import csv
import re

from .validation import ensure_extension, validate_filename

_SECTION_PATTERN = re.compile(r"^\[(?P<name>[^\]]+)\]$")
_HEADER_PATTERN = re.compile(r"^#\s*(?P<header>.+)$")
_COMMENT_PATTERN = re.compile(r"\s*//.*$")

def load_csv(filename: str, section: str | None = None) -> pd.DataFrame:
    """
    Load a custom CSV file with optional [sections] and '#' headers.
    Values are automatically converted to float if possible.
    """
    filename = validate_filename(filename, ".csv")

    data = []
    headers = None
    in_section = section is None  # True if no section specified

    with open(filename, newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue  # skip empty lines

            line = ",".join(row).strip()
            line = _remove_comments(line);
            if not line:
                continue; # skip comment only lines

            # Section handling
            matchSection = _SECTION_PATTERN.match(line)
            if matchSection and section is not None:
                in_section = (matchSection.group("name") == section)
                continue;
            if not in_section:
                continue;

            # Header row
            matchHeader = _HEADER_PATTERN.match(line)
            if matchHeader and "," in matchHeader.group("header"):
                headers = [x.strip() for x in matchHeader.group("header").split(",")]
                continue

            # Data row → apply type conversion
            row_clean = [x.strip() for x in line.split(",")]
            data.append([_maybe_number(x) for x in row_clean])

    return pd.DataFrame(data, columns=headers if headers else None)

def load_csv_datacluster(filename: str, section: str | None = None):
    from ..structs import DataCluster
    return DataCluster(load_csv(filename, section))

def load_csv_consts(filename, section: str | None = None) -> dict:
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
        print("Warning! No 'name' index found! Using indicies")
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

# ==================================================
#    helper
# ==================================================

def _maybe_number(x: str):
    """Try to convert to float, otherwise return original string."""
    try:
        return float(x)
    except ValueError:
        return x

def _remove_comments(line: str) -> str:
    """Remove comments from a line."""
    return _COMMENT_PATTERN.sub("", line).strip()

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

