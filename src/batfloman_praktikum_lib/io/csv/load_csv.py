import re
import csv
import pandas as pd

from ..validation import validate_filename

_SECTION_PATTERN = re.compile(r"^\[(?P<name>[^\]]+)\]$")
_HEADER_PATTERN = re.compile(r"^#\s*(?P<header>.+)$")
_COMMENT_PATTERN = re.compile(r"//.*$")

def load_csv(filename: str, section: str | None = None) -> pd.DataFrame:
    filename = validate_filename(filename, ".csv")

    data = []
    headers = None
    in_section = section is None  # True if no section specified
    section_found = section is None  # track if section ever appears

    with open(filename, newline="") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
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
                if in_section:
                    section_found = True
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

            if headers:
                expected = len(headers)
            else:
                expected = len(data[0]) if data else len(row_clean)

            if len(row_clean) < expected:
                print(f"Warning: row in line {i+1} has only {len(row_clean)} columns, expected {expected}: {row_clean}")

            data.append([_maybe_number(x) for x in row_clean])

    if section is not None and not section_found:
        raise ValueError(f"Section '{section}' not found in {filename}")

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
