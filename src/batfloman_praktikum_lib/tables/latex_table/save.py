import numpy as np;
import pandas as pd
import os

from typing import Optional

from ..validation import ensure_extension

from .formatter import get_column_format, format_header, format_value
from ..metadata import MetadataManager

# ==================================================

def export_as_latex_table(
    df: pd.DataFrame,
    path: str,
    header: bool = True, # include header row
    index: bool = False, # include DataFrame index as first column
    metadata_manager: Optional[MetadataManager] = None, 
) -> None:
    # ensure path settings
    path = ensure_extension(path, ".tex")
    dir_path = os.path.dirname(path)
    if dir_path:
        # TODO: ask user to create the dir first?
        os.makedirs(dir_path, exist_ok=True)

    # Build formatted headers if metadata is provided
    if metadata_manager and header:
        formatted_headers = [
            format_header(col, metadata_manager.get_metadata(col)) for col in df.columns
        ]
    else:
        formatted_headers = header  # can be True/False

    column_format = get_column_format(df.columns, metadata_manager)

    with open(path, 'w') as f:
        f.write("\\begin{tabular}{"+ column_format +"}\n")
        f.write("\t\\toprule\n")

        if header:
            f.write("\t" + " & ".join(formatted_headers) + " \\\\\n")
            f.write("\t\\midrule\n")

        for _, row in df.iterrows():
        # for i, row in enumerate(df.to_numpy()):
            # processed_row = [format_latex_string(str(v)) for v in row]
            processed_row = [format_value(v, metadata_manager.get_metadata(k)) for k, v in zip(df.columns, row)]
            f.write("\t" + " & ".join(processed_row) + " \\\\\n")
        
        f.write("\t\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Successfully exported table {os.path.basename(path)} to {os.path.abspath(path)}")

# ==================================================

# def export_as_latex(
#     arr: np.ndarray, 
#     path: str, 
#     has_header: bool = True,
#     metadata_manager: Optional[MetadataManager] = None, 
# ) -> None:
#     path = ensure_extension(path, ".tex")
#     rows, cols = arr.shape

#     dir_path = os.path.dirname(path)
#     if dir_path:
#         os.makedirs(dir_path, exist_ok=True)

#     with open(path, 'w') as f:
#         f.write("\\begin{tabular}{" + "c" * cols + "}\n")
#         f.write("\t\\toprule\n")
        
#         for i, row in enumerate(arr):
#             processed_row = [format_latex_string(str(v)) for v in row]
#             f.write("\t" + " & ".join(processed_row) + " \\\\\n")
#             if has_header and i == 0:
#                 f.write("\t\\midrule\n")
        
#         f.write("\t\\bottomrule\n")
#         f.write("\\end{tabular}\n")

#     print(f"Successfully exported table {os.path.basename(path)} to {os.path.abspath(path)}")

# latex_patterns = {
#     r'\bphi\b': r'$\varphi$',         # Replace 'phi' with $\varphi$
#     r"_{?(\w+)}?": r"_{\1}",
#     r"\^{?(\w+)}?": r"^{\1}",
#     r"(\d*\.?\d+\(?\d*\))(:?$|\s)": r"$\1$",
#     r"(\d*\.?\d+\(?\d*\))e(-?\d+)?": r"$\1 \\times 10^{\2}$",
# }
#
# def replace_latex_patterns(input_string, replacements = latex_patterns):
#     for pattern, replacement in replacements.items():
#         input_string = re.sub(pattern, replacement, input_string)
#     return input_string
#
# def export_as_latex(arr: np.ndarray, filename: str, has_header=True):
#     filename = ensure_extension(filename, ".tex")
#
#     if isinstance(arr, np.ndarray):
#         rows, cols = arr.shape
#
#     with open(filename, 'w') as f:
#         f.write("\\begin{tabular}{" + "c" * cols + "}\n")
#         f.write("\t\\toprule\n")
#
#         is_header = has_header;
#         for row in arr:
#             processed_row = [
#                 replace_latex_patterns(str(value)) for value in row
#             ]
#             f.write("\t" + " & ".join(processed_row) + " \\\\\n")
#             if is_header:
#                 f.write("\t\\midrule\n");
#                 is_header = False;
#
#         f.write("\t\\bottomrule\n")
#         f.write("\\end{tabular}\n")
#
#     print(f"Successfully exported table {os.path.basename(filename)} to {os.path.abspath(filename)}")
#

