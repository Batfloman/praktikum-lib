import json
import os

from .measurement import Measurement

class Dataset:
    def __init__(self, measurements: dict = None):
        self.measurements = measurements or {};

    def copy_remove_index(self, index):
        if not index in self.measurements:
            return Dataset(self.measurements)
        new_measurements = {k: v for k, v in self.measurements.items() if k != index}
        return Dataset(new_measurements)

    def __getitem__(self, index) -> Measurement:
        return self.measurements[index];

    def __setitem__(self, index, value):
        self.measurements[index] = value;
    
    def __contains__(self, key):
        return key in self.measurements;

    def __str__(self):
        strings = [];
        for key, value in self.measurements.items():
            strings.append(f"{key}: {value}")
        return f'{", ".join(strings)}'

    def __iter__(self):
        return iter(self.measurements.values())

    def to_json(self) -> str:
        data = {}
        for key, value in self.measurements.items():
            if isinstance(value, Measurement):
                data[key] = {
                    "value": value.value,
                    "error": value.error
                }
            else:
                data[key] = {"value": value}
        return json.dumps(data)

    @staticmethod
    def from_json(json_str: str):
        raw = json.loads(json_str)
        measurements = {}

        for key, entry in raw.items():
            if "error" in entry:
                measurements[key] = Measurement(entry["value"], entry["error"])
            else:
                measurements[key] = entry["value"]

        return Dataset(measurements)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()
        return Dataset.from_json(json_str)
