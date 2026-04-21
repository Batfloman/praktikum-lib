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
