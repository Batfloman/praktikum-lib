import pandas as pd
import os
import numpy as np
import csv

def _append_extension(filename:str, extension: str) -> str:
    # Ensures the filename has the correct extension
    if not filename.endswith(extension):
        filename += extension;
    return filename;

def _validate_filename(filename: str, extension: str) -> str:
    # Validates the existence of the file and appends the correct extension
    filename = _append_extension(filename, extension)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    return filename;

class load:
    @staticmethod
    def excel(filename: str) -> pd.DataFrame:
        # Loads data from an Excel file
        filename = _validate_filename(filename, ".xlsx")
        return pd.read_excel(filename);

    def csv(filename: str, section: str = None) -> pd.DataFrame:
        filename = _validate_filename(filename, ".csv")

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
                        headers = line[1:].strip().split(",")  # Remove `#` and split
                    else:
                        data.append(line.split(","))

        return pd.DataFrame(data, columns=headers if headers else None)
    
    @staticmethod
    def cassy(filename: str) -> pd.DataFrame:
        filename = _validate_filename(filename, ".txt")

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

class export:
    @staticmethod
    def excel(data, filename: str = "output"):
        # Exports data to an Excel file
        filename = _append_extension(filename, ".xlsx")

        try:
            if isinstance(data, np.ndarray):
                if data.ndim != 2:
                    raise ValueError("Only 2D numpy arrays are supported.")
                data = pd.DataFrame(data)
            elif isinstance(data, dict):
                data = pd.DataFrame(data)
            elif not isinstance(data, pd.DataFrame):
                raise ValueError(f"Cannot convert input to pd.DataFrame: {type(data)}")

            # Save the DataFrame to an Excel file
            data.to_excel(filename, index=False, header=True)
        except Exception as e:
            print(f"Error occurred: {e}")

    @staticmethod
    def csv(data, filename: str="output"):
        # Exports data to a CSV file
        filename = _append_extension(filename, ".csv");

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
            writer.writerows(data)
        except Exception as e:
            print(f"Error occurred: {e}")