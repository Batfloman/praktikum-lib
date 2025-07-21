import sympy as sp;

from .handle_sympy import to_sympy
    
def get_derivative(func, var):
    """
    Computes the derivative of a Python function or SymPy expression.

    Parameters:
    - func: Python function or SymPy expression to differentiate.
    - var: Variable with respect to which the derivative is taken. Can be a string or a SymPy symbol.
    - symbol_overrides: Optional overrides for argument symbols.

    Returns:
    - The derivative as a SymPy expression.
    """
    # Ensure `func` is a SymPy expression
    if not isinstance(func, sp.Basic):
        func = to_sympy(func)

    # Ensure `var` is a SymPy symbol
    if isinstance(var, str):
        var = sp.symbols(var)

    # Compute the derivative
    return sp.diff(func, var)

def gaussian_error_propagation(func, exclude=None, **symbol_overrides):
    """
    Computes the general form of Gaussian error propagation symbolically.

    Parameters:
    - func: Python function or SymPy expression for which uncertainty is propagated.
    - exclude: Optional set of symbols to exclude from propagation.
    - symbol_overrides: Optional overrides for argument symbols.

    Returns:
    - The general form of the propagated uncertainty as a SymPy expression.
    """
    # Ensure the function is a SymPy expression
    if not isinstance(func, sp.Basic):
        func = to_sympy(func, **symbol_overrides)

    # Determine free symbols in the function
    variables = func.free_symbols

    # Apply exclusions if provided
    if exclude is not None:
        if not isinstance(exclude, set):
            exclude = set(exclude)
        exclude = {sp.symbols(sym) if isinstance(sym, str) else sym for sym in exclude}
        variables = variables - exclude

    # General form of propagated uncertainty
    propagated = 0
    for symbol in variables:
        sigma = sp.symbols(f"\\sigma_{{{symbol}}}")  # Symbolic uncertainty for each variable
        derivative = sp.diff(func, symbol)
        propagated += (derivative * sigma) ** 2

    return sp.sqrt(propagated)