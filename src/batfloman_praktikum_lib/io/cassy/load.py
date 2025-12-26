import numpy as np
import pandas as pd

from ..validation import validate_filename

def load_cassy(filename: str) -> pd.DataFrame:
    filename = validate_filename(filename, ".txt")

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Locate header and data start
    data_start_index = 0
    header = None
    for i, line in enumerate(lines):
        if "DEF=" in line:
            header = line.strip().replace('"', '').replace("DEF=", "").split("\t")
            data_start_index = i + 1  # Data starts after the header line
            break  # Stop searching after finding the header

    # Process data
    data = []
    for line in lines[data_start_index:]:
        values = line.replace(",", ".").strip().split("\t")
        data.append([float(v) if v != "NAN" else np.nan for v in values])  # Convert to float, handle "NAN"

    # Convert to DataFrame
    return pd.DataFrame(data, columns=header if header else None)
