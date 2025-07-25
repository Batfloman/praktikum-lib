import numpy as np;
import re;
import os;
import pandas as pd

from typing import Union

latex_patterns = {
    r'\bphi\b': r'$\varphi$',         # Replace 'phi' with $\varphi$
    r"_{?(\w+)}?": r"_{\1}",
    r"\^{?(\w+)}?": r"^{\1}",
    r"(\d*\.?\d+\(?\d*\))(:?$|\s)": r"$\1$",
    r"(\d*\.?\d+\(?\d*\))e(-?\d+)?": r"$\1 \\times 10^{\2}$",
}

def replace_latex_patterns(input_string, replacements = latex_patterns):
    for pattern, replacement in replacements.items():
        input_string = re.sub(pattern, replacement, input_string)
    return input_string

def save_table(arr: np.ndarray, filename: str, has_header=True):
    if not filename.endswith(".tex"):
        filename = filename + ".tex";

    if isinstance(arr, np.ndarray):
        rows, cols = arr.shape

    with open(filename, 'w') as f:
        f.write("\\begin{tabular}{" + "c" * cols + "}\n")
        f.write("\t\\toprule\n")
        
        is_header = has_header;
        for row in arr:
            processed_row = [
                replace_latex_patterns(str(value)) for value in row
            ]
            f.write("\t" + " & ".join(processed_row) + " \\\\\n")
            if is_header:
                f.write("\t\\midrule\n");
                is_header = False;
        
        f.write("\t\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    print(f"Successfully exported table {os.path.basename(filename)} to {os.path.abspath(filename)}")