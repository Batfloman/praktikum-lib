import json
import os
from typing import Any

from ...path_managment import ensure_extension
from ...structs.dataCluster import DataCluster
from ...structs.dataset import Dataset
from ...structs.measurement import Measurement


def from_json_data(data: Any) -> Any:
    if isinstance(data, list):
        return [from_json_data(item) for item in data]

    if not isinstance(data, dict):
        return data

    obj_type = data.get("__type__")
    if obj_type == "Measurement":
        return Measurement.from_dict(data)
    if obj_type == "Dataset":
        return Dataset.from_dict(data)
    if obj_type == "DataCluster":
        return DataCluster.from_dict(data)

    return {key: from_json_data(value) for key, value in data.items()}


def loads_json(json_str: str) -> Any:
    return from_json_data(json.loads(json_str))


def load_json(path: str | os.PathLike) -> Any:
    path = os.fspath(path)
    path = ensure_extension(path, ".json")

    with open(path, "r", encoding="utf-8") as file:
        return loads_json(file.read())
