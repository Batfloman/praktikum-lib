import re;
from typing import Union, Callable;
from sympy.parsing.latex import parse_latex
import sympy as sp;
import os;

from .handle_sympy import to_sympy, from_sympy;

def import_latex_as_sympy(filename: str) -> sp.Expr:
    with open(filename, "r") as file:
        latex_content = file.read();

    extracted_eq = _extract_equation(latex_content)
    if extracted_eq:
        expr = sp.sympify(parse_latex(extracted_eq))
        if isinstance(expr, sp.Eq):
            return expr.rhs;
        return expr;
    else:
        raise ValueError("No valid equation found in the input.")

def import_latex_as_func(filename: str) -> Callable:
    return from_sympy(import_latex_as_sympy(filename));

def export_as_latex(func, filename: str, **symbol_overrides) -> None:
    """
    Converts a function or SymPy expression into a LaTeX representation and saves it to a file.

    Parameters:
    - func: Python function or SymPy expression to convert.
    - filename: Output filename for the LaTeX file.
    - symbol_overrides: Optional overrides for argument symbols.

    Raises:
    - ValueError: If `func` is not a valid Python function or SymPy expression.
    """
    # Ensure the function is a SymPy expression
    if not isinstance(func, sp.Basic):
        try:
            func = to_sympy(func, **symbol_overrides)
        except Exception as e:
            raise ValueError("Invalid input: unable to convert to SymPy.") from e

    # Add .tex extension if not provided
    if not filename.endswith(".tex"):
        filename += ".tex"

    absolute_path = os.path.abspath(filename)
    file_name = os.path.basename(absolute_path)

    # Write the LaTeX string to a file
    try:
        with open(filename, 'w') as f:
            f.write(sp.latex(func))
        print(f"Exported equation to {file_name} at {absolute_path}")
    except IOError as e:
        raise IOError(f"Failed to write LaTeX to file: {absolute_path}") from e

def _extract_equation(latex_str) -> Union[str, None]:
    match = re.search(r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}", latex_str, re.DOTALL)
    equation = match.group(1).strip();    
    if not equation:
        return None;

    equation = re.sub(r"\\tag\{.*?\}", "", equation)
    equation = re.sub(r"\\label\{.*?\}", "", equation)
    return equation;