from scipy.optimize import root_scalar, minimize_scalar
import numpy as np

def find_max(fit_func, x_range):
    """
    Find the maximum value of a function within a given range.

    Parameters:
    fit_func (callable): The function to find the maximum of.
    x_range (tuple, optional): The range (start, end) within which to search for the maximum.
                               Defaults to None, which corresponds to the entire real line.

    Returns:
    max_x (float): The x-coordinate of the maximum point.
    max_val (float): The maximum value of the function.
    """
    result = minimize_scalar(lambda x: -fit_func(x), bounds=x_range, method='bounded')
    max_x = result.x
    max_val = -result.fun  # Note: Since we minimized -fit_func, we need to negate the result to get the maximum value
    return max_x, max_val

def find_x_for_y(fit_func, target_y, x_range, num_points=1000, tol=1e-5):
    """
    Find the x values for which the function evaluates to a specific y value.

    Parameters:
    fit_func (callable): The function to evaluate.
    target_y (float): The target y value to find corresponding x values for.
    x_range (tuple): The range (start, end) within which to search for x values.
    num_points (int, optional): Number of points to divide the range into for searching. Defaults to 1000.
    tol (float, optional): Tolerance for considering two x values as the same. Defaults to 1e-5.

    Returns:
    x_values (list): A list of x values for which fit_func(x) = target_y.
    """
    x_values = []
    x_min, x_max = x_range
    x_points = np.linspace(x_min, x_max, num_points)

    # Scan the interval in small segments
    for i in range(len(x_points) - 1):
        x0, x1 = x_points[i], x_points[i + 1]
        
        # Define the function to find the root of
        def func_to_solve(x):
            return fit_func(x) - target_y
        
        # Try to find a root in the interval [x0, x1]
        try:
            sol = root_scalar(func_to_solve, bracket=[x0, x1], method='brentq')
            if sol.converged:
                root = sol.root
                # Check if the root is not too close to already found roots
                if not any(np.isclose(root, existing_root, atol=tol) for existing_root in x_values):
                    x_values.append(root)
        except ValueError:
            # No root found in this interval
            continue
    
    return x_values
