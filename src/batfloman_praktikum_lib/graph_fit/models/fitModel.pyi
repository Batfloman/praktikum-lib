from typing import List, Optional
import numpy as np

from batfloman_praktikum_lib.graph_fit.init_params.order_init_params import InitalParamGuess
from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from ..fitResult import FIT_METHODS, FitResult
from .modelMeta import ModelMeta

class FitModel(metaclass=ModelMeta):
    @staticmethod
    def model(x: np.ndarray, *args, **kwargs) -> float: ...
    
    @staticmethod
    def get_initial_guess(x: np.ndarray, y: np.ndarray) -> List[float]: ...
    
    @staticmethod
    def get_param_names() -> List[str]: ...
    
    @classmethod
    def fit(
        cls, 
        x: np.ndarray, 
        y: np.ndarray, 
        xerr: Optional[np.ndarray] = None, 
        yerr: Optional[np.ndarray] = None,
        *,
        initial_guess: Optional[InitalParamGuess] = None,
        method: Optional[FIT_METHODS] = None,
    ) -> FitResult: ...
    
    @classmethod
    def ls_fit(
        cls, 
        x: np.ndarray, 
        y: np.ndarray, 
        yerr: Optional[np.ndarray] = None,
        *,
        initial_guess: Optional[InitalParamGuess] = None,
        ignore_warning_x_errors: bool = False,
        ignore_warning_y_errors: bool = False,
    ) -> FitResult: ...
    
    @classmethod
    def odr_fit(
        cls, 
        x: np.ndarray, 
        y: np.ndarray, 
        xerr: Optional[np.ndarray] = None, 
        yerr: Optional[np.ndarray] = None,
        *,
        initial_guess: Optional[InitalParamGuess] = None,
    ) -> FitResult: ...
    
    @classmethod
    def on_data(
        cls, 
        data: DataCluster, 
        x_index: str, 
        y_index: str,
        *,
        initial_guess: Optional[InitalParamGuess] = None,
        method: Optional[FIT_METHODS] = None,
    ) -> FitResult: ...

