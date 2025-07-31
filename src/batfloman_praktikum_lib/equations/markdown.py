from typing import Union, Callable
import sympy as sp;
import re;
from IPython.display import Markdown, display as _ipython_display

from .handle_sympy import to_sympy;

def display(input_data: Union[str, Callable, sp.Basic]) -> None:
    """
    Processes the input and displays a LaTeX-rendered equation in a Jupyter Notebook.
    
    Parameters:
    - input_data (Union[str, Callable, Basic]): 
        - str: Path to a LaTeX file.
        - Callable: A Python function to be converted to a sympy expression.
        - Basic: A sympy symbol/expression.
    """
    if isinstance(input_data, str):
        # Handle file path
        try:
            with open(input_data, 'r', encoding='utf-8') as f:
                latex_content = f.read()
            
            # Validate and clean LaTeX content
            cleaned_content = re.sub(r"\\label\{[^}]*\}", "", latex_content)
            _ipython_display(Markdown(f"$$ {cleaned_content} $$"))
        except FileNotFoundError:
            raise ValueError(f"The file at path '{input_data}' was not found.")
        except Exception as e:
            raise ValueError(f"Error reading the file: {e}")
    
    elif callable(input_data):
        # Handle function
        try:
            sympy_expr = to_sympy(input_data)
            latex_expr = sp.latex(sympy_expr)
            _ipython_display(Markdown(f"$$ {latex_expr} $$"))
        except Exception as e:
            raise ValueError(f"Error converting function to sympy expression: {e}")
    
    elif isinstance(input_data, sp.Basic):
        # Handle sympy expression
        latex_expr = sp.latex(input_data)
        _ipython_display(Markdown(f"$$ {latex_expr} $$"))
    
    else:
        raise TypeError("Input must be a file path (str), a callable function, or a sympy expression.")