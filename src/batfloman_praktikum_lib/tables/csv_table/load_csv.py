import re
import csv
import pandas as pd

from ..validation import validate_filename

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

def _maybe_number(x: str):
    """Try to convert to float, otherwise return original string."""
    try:
        return float(x)
    except ValueError:
        return x

def _remove_comments(line: str) -> str:
    """Remove comments from a line."""
    return _COMMENT_PATTERN.sub("", line).strip()
