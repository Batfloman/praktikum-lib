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

    def __rmul__(cls, other: int) -> Type[CompositeFitModel]:
        """
        Repeat a FitModel class multiple times to create a composite model.

        Example usage:
            Repeated = 3 * Gaussian
            result = Repeated.fit(x_data, y_data)

        Args:
            other (int): Number of repetitions of the model class.

        Returns:
            Type[CompositeFitModel]: A new composite model class with repeated components.
        """
        ...

    def __mul__(cls, other: int) -> Type[CompositeFitModel]:
        """
        Allows multiplication from the left: `Gaussian * 3`.

        Internally delegates to `__rmul__` to avoid code duplication.

        Example usage:
            Repeated = Gaussian * 3
            result = Repeated.fit(x_data, y_data)

        Args:
            other (int): Number of repetitions of the model class.

        Returns:
            Type[CompositeFitModel]: A new composite model class with repeated components.
        """
        ...
