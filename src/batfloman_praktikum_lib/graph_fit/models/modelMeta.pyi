from typing import Type, Union
from .fitModel import FitModel
from .compositeFitModel import CompositeFitModel

class ModelMeta(type):
    def __add__(
        cls, 
        other: Type[Union[FitModel, CompositeFitModel]]
    ) -> Type[CompositeFitModel]:
        """
        Combine two FitModel classes (or already `CompositeFitModels`) into a new static composite model.

        Example usage:
            Combined = Gaussian + Linear
            popt, pcov = curve_fit(Combined, x_data, y_data, p0=Combined.get_initial_guess(x_data, y_data))

        Returns a subclass to `CompositeFitModel(FitModel)` with static methods:
            - model(x, *params)
            - get_param_names()
            - get_initial_guess(x, y)
        """
        ...
