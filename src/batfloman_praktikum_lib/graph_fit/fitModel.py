from typing import List

from structs import DataCluster
from .least_squares import generic_fit as ls_fit
from .orthogonal_distance import generic_fit as odr_fit
from .fitResult import FitResult

class FitModel:
    @staticmethod
    def model(x, **args) -> float:
        raise NotImplementedError("Subclasses must implement the model method.")
    
    @staticmethod
    def get_initial_guess(x, y) -> List[float]:
        raise NotImplementedError("Subclasses must implement the model method.")

    @staticmethod
    def get_param_names() -> List[str]:
        raise NotImplementedError("Subclasses must implement the model method.")

    @classmethod
    def fit(cls, x, y, xerr = None, yerr = None) -> FitResult:
        return cls.ls_fit(x, y, yerr=yerr);
        # has_xerr = xerr is not None and xerr.size > 0;
        # if has_xerr:
        #     return cls.odr_fit(x, y, xerr=xerr, yerr=yerr);
        # else:
        #     return cls.ls_fit(x, y, yerr=yerr);
    
    @classmethod
    def ls_fit(cls, x, y, yerr = None) -> FitResult:
        param_names = cls.get_param_names()
        initial_guess = cls.get_initial_guess(x, y)
        return ls_fit(cls.model, x, y, yerr, initial_guess, param_names)
    
    @classmethod
    def odr_fit(cls, x, y, xerr = None, yerr = None) -> FitResult:
        param_names = cls.get_param_names()
        initial_guess = cls.get_initial_guess(x, y)
        return odr_fit(cls.model, x, y, xerr=xerr, yerr=yerr, initial_guess=initial_guess, param_names=param_names)
    
    @classmethod
    def on_data(cls, data: DataCluster, x_index: str, y_index: str) -> FitResult:
        if not x_index in data.get_column_names():
            raise NameError(f"Column {x_index} not found")
        if not y_index in data.get_column_names():
            raise NameError(f"Column {y_index} not found")

        x_values = data.values(x_index);
        y_values = data.values(y_index);
        x_errors = data.errors(x_index);
        y_errors = data.errors(y_index);

        return cls.fit(x_values, y_values, xerr=x_errors, yerr=y_errors);
