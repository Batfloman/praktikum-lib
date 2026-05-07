import json
import os
from typing import Any

from ...path_managment import create_dirs, ensure_extension


def to_json_data(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [to_json_data(item) for item in obj]
    if isinstance(obj, tuple):
        return [to_json_data(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_json_data(value) for key, value in obj.items()}
    return obj


def dumps_json(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_json_data(obj), indent=indent)


def save_json(
    obj: Any,
    path: str | os.PathLike,
    *,
    indent: int | None = 2,
) -> str:
    path = os.fspath(path)
    path = ensure_extension(path, ".json")
    create_dirs(path)

    with open(path, "w", encoding="utf-8") as file:
        file.write(dumps_json(obj, indent=indent))

    return path
