from collections.abc import Iterable, Iterator, Mapping

class Dataset:
    def __init__(self, measurements: Mapping | None = None):
        self.measurements = dict(measurements or {})

    def copy(self) -> "Dataset":
        return Dataset(self.measurements)

    def copy_remove_index(self, index):
        if index not in self.measurements:
            return Dataset(self.measurements)
        new_measurements = {k: v for k, v in self.measurements.items() if k != index}
        return Dataset(new_measurements)

    def without(self, *indices) -> "Dataset":
        filtered = {k: v for k, v in self.measurements.items() if k not in indices}
        return Dataset(filtered)

    def select(self, *indices) -> "Dataset":
        selected = {k: self.measurements[k] for k in indices if k in self.measurements}
        return Dataset(selected)

    def rename(self, **mapping) -> "Dataset":
        renamed = {mapping.get(k, k): v for k, v in self.measurements.items()}
        return Dataset(renamed)

    def __getitem__(self, index):
        return self.measurements[index]

    def __setitem__(self, index, value):
        self.measurements[index] = value

    def __delitem__(self, index):
        del self.measurements[index]

    def __contains__(self, key):
        return key in self.measurements

    def __str__(self):
        strings = []
        for key, value in self.measurements.items():
            strings.append(f"{key}: {value}")
        return f'{", ".join(strings)}'

    def __repr__(self):
        return f"Dataset({self.measurements!r})"

    def __iter__(self) -> Iterator:
        return iter(self.measurements)

    def __len__(self) -> int:
        return len(self.measurements)

    # ==================================================

    def keys(self):
        return self.measurements.keys()

    def values(self):
        return self.measurements.values()

    def items(self):
        return self.measurements.items()

    def get(self, key, default=None):
        return self.measurements.get(key, default)

    def update(self, other: Mapping | Iterable[tuple] = (), **kwargs) -> None:
        self.measurements.update(other, **kwargs)

    def to_dict(self) -> dict:
        serialized = {}
        for key, value in self.measurements.items():
            if hasattr(value, "to_dict"):
                serialized[key] = value.to_dict()
            else:
                serialized[key] = value

        return {
            "__type__": "Dataset",
            "measurements": serialized,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        from .measurement import Measurement

        measurements = {}
        for key, value in data["measurements"].items():
            if isinstance(value, dict) and value.get("__type__") == "Measurement":
                measurements[key] = Measurement.from_dict(value)
            else:
                measurements[key] = value

        return cls(measurements)

    def to_json(self, *, indent: int | None = 2) -> str:
        from batfloman_praktikum_lib.io.json import dumps_json
        return dumps_json(self, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Dataset":
        from batfloman_praktikum_lib.io.json import loads_json
        return loads_json(json_str)

    def save_json(self, path: str, *, indent: int | None = 2) -> str:
        from batfloman_praktikum_lib.io.json import save_json
        return save_json(self, path, indent=indent)

    @classmethod
    def load_json(cls, path: str) -> "Dataset":
        from batfloman_praktikum_lib.io.json import load_json
        return load_json(path)
