import numpy as np;
import os

from .formatter import format_latex_string
from ..validation import ensure_extension

def export_as_latex(arr: np.ndarray, path: str, has_header: bool = True) -> None:
    path = ensure_extension(path, ".tex")
    rows, cols = arr.shape

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(path, 'w') as f:
        f.write("\\begin{tabular}{" + "c" * cols + "}\n")
        f.write("\t\\toprule\n")
        
        for i, row in enumerate(arr):
            processed_row = [format_latex_string(str(v)) for v in row]
            f.write("\t" + " & ".join(processed_row) + " \\\\\n")
            if has_header and i == 0:
                f.write("\t\\midrule\n")
        
        f.write("\t\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Successfully exported table {os.path.basename(path)} to {os.path.abspath(path)}")

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

