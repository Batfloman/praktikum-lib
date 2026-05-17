import numpy as np;

from .fitModel import FitModel

class ConstFunc(FitModel):
    @staticmethod
    def model(x, b):
        return b;
    
    @staticmethod
    def get_param_names():
        return ["b"]

    @staticmethod
    def get_initial_guess(x, y):
        b = np.mean(y)
        return [b]

class Linear(FitModel):
    @staticmethod
    def model(x, m, n):
        return m * x + n

    @staticmethod
    def get_param_names():
        return ["m", "n"]

    @staticmethod
    def get_initial_guess(x, y):
        start_idx = np.argmin(x)
        end_idx = np.argmax(x)
        dy = y[end_idx] - y[start_idx]
        dx = x[end_idx] - x[start_idx]
        m = dy / dx if dx != 0 else 0
        n = y[start_idx] - m * x[start_idx]
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

class AmpTiefpass(FitModel):
    @staticmethod
    def model(f, A0, fc):
        # Betrag der Verstärkung
        return A0 / np.sqrt(1 + (f/fc)**2)

    @staticmethod
    def get_param_names():
        return ["A0", "f_grenz"]

    @staticmethod
    def get_initial_guess(x, y):
        A0 = np.max(y)
        # fc ~ Frequenz bei -3dB (A0/sqrt(2))
        half = A0 / np.sqrt(2)
        # finde Index, wo y ~ half
        idx = np.argmin(np.abs(y - half))
        fc = x[idx] if len(x) > 0 else 1.0
        return [A0, fc]

class Gaussian(FitModel):
    @staticmethod
    def model(x, A, sigma, x0):
        return A * np.exp(- (x-x0)**2 / (2*sigma**2) )

    @staticmethod
    def get_param_names():
        return ["A", "sigma", "x0"]

    @staticmethod
    def get_initial_guess(x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        # estimate and subtract baseline
        baseline = np.percentile(y, 10)
        yw = y - baseline
        yw = np.clip(yw, 0, None)

        # amplitude
        A = np.max(y) - baseline

        # center: position of maximum is usually more robust than weighted mean
        x0 = x[np.argmax(y)]

        # sigma estimate from FWHM
        half_max = baseline + A / 2
        above = y >= half_max

        if np.sum(above) >= 2:
            x_left = x[above][0]
            x_right = x[above][-1]
            fwhm = x_right - x_left
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            # fallback: weighted variance
            if np.sum(yw) > 0:
                sigma = np.sqrt(np.sum(yw * (x - x0)**2) / np.sum(yw))
            else:
                sigma = (np.max(x) - np.min(x)) / 6

        # avoid zero/negative sigma
        sigma = max(sigma, np.finfo(float).eps)

        return [A, sigma, x0]
