import ast
import inspect
import numpy as np;
import sympy as sp;
import textwrap
import sys

def from_sympy(expr):
    arg_names = [str(sym).replace("{", "").replace("}", "").replace("\\", "").replace(" ", "_") for sym in expr.free_symbols]
    replacements = {sym: sp.Symbol(arg_name) for sym, arg_name in zip(expr.free_symbols, arg_names)}
    
    expr = expr.subs(replacements)  # Substitute the new symbols
    return sp.lambdify(arg_names, expr, modules="sympy")

def to_sympy(func):
    if isinstance(func, sp.Basic):  # Already a SymPy expression
        return sp.simplify(func)
    
    source = inspect.getsource(func)
    source = textwrap.dedent(source)  # Remove indentation
    tree = ast.parse(source)  # Parse function into an AST
    
    func_def = tree.body[-1]  # Extract function definition
    symbols = _extract_symbols(func_def)
    closure_vars = _extract_closure_vars(func)
    
    local_dict = symbols.copy()
    for node in func_def.body:
        if isinstance(node, ast.Assign):  # Variable assignment, like a = ...
            var_name = node.targets[-1].id
            var_value = ast.unparse(node.value)
            var_value = _replace_function_calls(var_value, local_dict)  # Replace function calls
            local_dict[var_name] = sp.sympify(var_value, locals=local_dict)
        elif isinstance(node, ast.Return):  # Return statement
            return_expr = ast.unparse(node.value)
            return_expr = _replace_function_calls(return_expr, local_dict)  # Replace function calls in return
            return sp.sympify(return_expr, locals={**local_dict, **closure_vars});

def _extract_symbols(func_def):
    args = [arg.arg for arg in func_def.args.args]  # Extract argument names
    return {arg: sp.Symbol(arg) for arg in args}  # Convert args to SymPy symbols

def _extract_closure_vars(func):
    closure_vars = {}
    if func.__closure__:
        free_vars = func.__code__.co_freevars
        closure_vals = [cell.cell_contents for cell in func.__closure__]
        for var, val in zip(free_vars, closure_vals):
            if callable(val):  # If it's a function, try converting it
                closure_vars[var] = to_sympy(val)
            else:
                try:
                    closure_vars[var] = sp.sympify(val)
                except sp.SympifyError:
                    print(f"Warning: Could not sympify closure variable '{var}' of type {type(val)}.")
                    pass  # Ignore non-sympifiable values
    return closure_vars

NUMPY_TO_SYMPY = {
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "sqrt": sp.sqrt,
}

def _replace_function_calls(expr, local_dict):
    tree = ast.parse(expr)
    
    def process_call(subnode):
        if isinstance(subnode.func, ast.Name):  # Direct function call by name
            func_name = subnode.func.id
            
            # Check global scope of calling script
            caller_globals = sys.modules['__main__'].__dict__
            func_ref = caller_globals.get(func_name) or globals().get(func_name)

            if callable(func_ref):
                func_args = [sp.sympify(ast.unparse(arg), locals=local_dict) for arg in subnode.args]
                func_expr = to_sympy(func_ref)  # Convert function
                return func_expr.subs(dict(zip(func_ref.__code__.co_varnames, func_args)))
        
        elif isinstance(subnode.func, ast.Attribute) and isinstance(subnode.func.value, ast.Name):
            module_name = subnode.func.value.id
            func_name = subnode.func.attr
            if module_name == "np" and func_name in NUMPY_TO_SYMPY:
                func_args = [sp.sympify(ast.unparse(arg), locals=local_dict) for arg in subnode.args]
                return NUMPY_TO_SYMPY[func_name](*func_args)
        
        return None
    
    new_expr = expr
    for subnode in ast.walk(tree):
        if isinstance(subnode, ast.Call):
            replacement = process_call(subnode)
            if replacement is not None:
                new_expr = new_expr.replace(ast.unparse(subnode), str(replacement))
    
    return new_expr