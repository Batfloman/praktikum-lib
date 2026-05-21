from typing import List, Optional, Mapping, Any

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from ..fitResult import FitResult, FIT_METHODS
from .modelMeta import ModelMeta
from abc import ABC, abstractmethod

import numpy as np

class FitModel(ABC, metaclass=ModelMeta): # type: ignore[misc]
    @staticmethod
    @abstractmethod
    def model(x, *args, **kwargs) -> float:
        pass
        # raise NotImplementedError("Subclasses must implement the model method.")
    
    @staticmethod
    @abstractmethod
    def get_initial_guess(x, y) -> List[float]:
        pass
        # raise NotImplementedError("Subclasses must implement the model method.")

    @staticmethod
    @abstractmethod
    def get_param_names() -> List[str]:
        pass
        # raise NotImplementedError("Subclasses must implement the model method.")

    @classmethod
    def fit(cls, x, y, xerr = None, yerr = None,
        *,
        initial_guess = None,
        fixed_params: Mapping[str, Any] | None = None,
        method: Optional[FIT_METHODS] = None,
    ) -> FitResult:
        from batfloman_praktikum_lib.structs.measurementBase import MeasurementBase
        has_xerr = xerr is not None and xerr.size > 0;
        x_has_err = any(isinstance(val, MeasurementBase) and val.error is not None for val in x)

        if method == "least squares":
            return cls.ls_fit(x, y, yerr=yerr, initial_guess=initial_guess, fixed_params=fixed_params, ignore_warning_x_errors=True);
        if method == "ODR":
            return cls.odr_fit(x, y, xerr=xerr, yerr=yerr, initial_guess=initial_guess, fixed_params=fixed_params);

        if has_xerr or x_has_err:
            return cls.odr_fit(x, y, xerr=xerr, yerr=yerr, initial_guess=initial_guess, fixed_params=fixed_params);
        else:
            return cls.ls_fit(x, y, yerr=yerr, initial_guess=initial_guess, fixed_params=fixed_params);
    
    @classmethod
    def ls_fit(cls, x, y, yerr = None,
        *,
        initial_guess = None,
        fixed_params: Mapping[str, Any] | None = None,
        ignore_warning_x_errors: bool = False,
        ignore_warning_y_errors: bool = False,
    ) -> FitResult:
        from ..least_squares import generic_fit

        if all(hasattr(y, "value") and hasattr(y, "error") for y in y):
            # Extract values and errors
            yerr = np.array([y.error for y in y])
            y = np.array([y.value for y in y])

        param_names = cls.get_param_names()
        initial_guess = initial_guess if (initial_guess is not None) else cls.get_initial_guess(x, y)

        res = generic_fit(cls.model, x, y, yerr,
            initial_guess=initial_guess, 
            fixed_params=fixed_params,
            param_names=param_names,
            ignore_warning_x_errors=ignore_warning_x_errors,
            ignore_warning_y_errors=ignore_warning_y_errors,
        )

        return res
    
    @classmethod
    def odr_fit(cls, x, y, xerr = None, yerr = None, 
        *,
        initial_guess = None,
        fixed_params: Mapping[str, Any] | None = None,
    ) -> FitResult:
        from ..orthogonal_distance import generic_fit as odr_fit

        param_names = cls.get_param_names()
        initial_guess = initial_guess if (initial_guess is not None) else cls.get_initial_guess(x, y)
        return odr_fit(cls.model, x, y, x_err=xerr, y_err=yerr, initial_guess=initial_guess, fixed_params=fixed_params, param_names=param_names)
    
    @classmethod
    def on_data(cls, data: DataCluster, x_index: str, y_index: str,
        *,
        initial_guess = None,
        fixed_params: Mapping[str, Any] | None = None,
        method: Optional[FIT_METHODS] = None,
    ) -> FitResult:
        if not x_index in data.get_column_names():
            raise NameError(f"Column {x_index} not found")
        if not y_index in data.get_column_names():
            raise NameError(f"Column {y_index} not found")

        x_values = data.column(x_index);
        y_values = data.column(y_index);

        return cls.fit(x_values, y_values, initial_guess=initial_guess, fixed_params=fixed_params, method=method);
