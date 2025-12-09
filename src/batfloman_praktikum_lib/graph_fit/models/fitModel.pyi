from typing import List, Optional
import numpy as np

from batfloman_praktikum_lib.structs.dataCluster import DataCluster
from ..fitResult import FitResult
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
        yerr: Optional[np.ndarray] = None
    ) -> FitResult: ...
    
    @classmethod
    def ls_fit(
        cls, 
        x: np.ndarray, 
        y: np.ndarray, 
        yerr: Optional[np.ndarray] = None
    ) -> FitResult: ...
    
    @classmethod
    def odr_fit(
        cls, 
        x: np.ndarray, 
        y: np.ndarray, 
        xerr: Optional[np.ndarray] = None, 
        yerr: Optional[np.ndarray] = None
    ) -> FitResult: ...
    
    @classmethod
    def on_data(
        cls, 
        data: DataCluster, 
        x_index: str, 
        y_index: str
    ) -> FitResult: ...

