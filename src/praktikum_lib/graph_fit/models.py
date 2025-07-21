import numpy as np;

from .fitModel import FitModel

class Linear(FitModel):
    @staticmethod
    def model(x, m, n):
        return m * x + n
    
    @staticmethod
    def get_param_names():
        return ["m", "n"]

    @staticmethod
    def get_initial_guess(x, y):
        dy = np.max(y) - np.min(y)
        dx = np.max(x) - np.min(x)
        m = dy / dx if dx != 0 else 0
        n = np.min(y) if m > 0 else np.max(y)
        return [m, n]

class Exponential(FitModel):
    @staticmethod
    def model(x, a, b, x0):
        return np.exp(a * (x-x0)) + b;
    
    @staticmethod
    def get_param_names():
        return ["a", "b", "x0"]
    
    @staticmethod
    def get_initial_guess(x, y):
        x0 = np.mean(x) # center
        y_shifted = y - np.min(y) + 1e-6  # Avoid log(0)
    
        log_y = np.log(y_shifted)  # Log transformation
        slope, intercept = np.polyfit(x - x0, log_y, 1)  # Fit straight line

        a = slope  # Since ln(y) ~ a * (x - x0)
        b = np.min(y) # starts at bottom y?
        return [a, b, x0]

class LimitedGrowth(FitModel):
    @staticmethod
    def model(x, a, b, max_value):
        return max_value - a * np.exp(b * x);

    @staticmethod
    def get_param_names():
        return ["a", "b", "max"]

    @staticmethod
    def get_initial_guess(x, y):
        a = np.max(y) - np.min(y)  # a ≈ max(y) - min(y)
        b = -1 / (np.max(x) - np.min(x))  # b ≈ -1 / range(x), assuming smooth decay
        max_value = np.max(y)  # max ≈ max(y)
        return [a, b, max_value]

class InverseSquare(FitModel):
    @staticmethod
    def model(x, a, b, c, x0):
        """Modified inverse-square model: y = a / ((x - x0)^2 + b) + c"""
        return a / ((x - x0)**2 + b) + c
    
    @staticmethod
    def get_param_names():
        return ["a", "b", "c", "x0"]

    @staticmethod
    def get_initial_guess(x, y):
        a = np.max(y)
        b = 1
        c = np.min(y)
        x0 = np.mean(x)
        return [a, b, c, x0]

class ResonanceCurve(FitModel):
    @staticmethod
    def model(x, a, x0, beta):
        denom = ((x0**2 - x**2)**2 + (2 * beta * x)**2)**0.5
        return a / denom 
    
    @staticmethod
    def get_param_names():
        return ["a", "x0", "beta"]

    @staticmethod
    def get_initial_guess(x, y):
        a = np.max(y)
        x0 = x[np.argmax(y)]
        beta = .1
        return [a, x0, beta]
