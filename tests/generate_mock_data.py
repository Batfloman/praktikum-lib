from dataclasses import dataclass
from typing import Optional, List, Union
import random

from batfloman_praktikum_lib.structs.dataset import Dataset
from batfloman_praktikum_lib import Measurement

def generate_random_measurement(
    mu: float=10,
    sigma: float=2,
    err_factor: float=0.05
):
    val = random.gauss(mu, sigma)              # normalverteilte Messwerte
    err = abs(random.gauss(0, err_factor*mu))  # zufälliger Fehler z.B. 5% Niveau
    return Measurement(val, err)

@dataclass
class MeasurementSetting:
    name: Optional[str] = None
    mu: float = 10
    sigma: float = 2
    err_factor: float = 0.05

def generate_random_dataset(
    measurement_settings: List[Union[MeasurementSetting, float]]
):
    d = {}
    for idx, setting in enumerate(measurement_settings):
        if isinstance(setting, MeasurementSetting):
            name = setting.name or f"param_{idx}"
            d[name] = generate_random_measurement(
                mu=setting.mu,
                sigma=setting.sigma,
                err_factor=setting.err_factor
            )
        else:
            name = f"param_{idx}"
            d[name] = setting
    return Dataset(d)

