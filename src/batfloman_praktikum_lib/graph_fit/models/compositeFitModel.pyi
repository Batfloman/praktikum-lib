from typing import List, Tuple, Type
from .fitModel import FitModel

class CompositeFitModel(FitModel):
    _components: List[Type[FitModel]]

    @staticmethod
    def model(x, **params) -> float: ...
    
    @staticmethod
    def get_param_names() -> List[str]: ...
    
    @staticmethod
    def get_initial_guess(x, y) -> List[float]: ...

def make_static_model_full(
    components: List[Type[FitModel]]
) -> Tuple[
    staticmethod,  # model(x, **params)
    staticmethod,  # get_param_names()
    staticmethod   # get_initial_guess(x, y)
]: ...

