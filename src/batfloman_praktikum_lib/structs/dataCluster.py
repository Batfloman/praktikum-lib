from dataclasses import asdict
import numpy as np
import pandas as pd
import re
import copy
from collections.abc import Mapping, Sequence
from typing import List, Union, Callable, Optional, Iterable

from batfloman_praktikum_lib.tables.latex_table import formatter as latex_formatter
from batfloman_praktikum_lib.structs.measurement import Measurement
from batfloman_praktikum_lib.structs.dataset import Dataset 

from ..tables.validation import ensure_extension

def _get_column_with_error_indicies(indicies: List[str]) -> dict:
    property_has_error = {}

    without_d = [x for x in indicies if not x.startswith("d")]
    with_d = [x for x in indicies if x.startswith("d")]
    error_pattern = r"d_?(.*)"

    for index in without_d:
        property_has_error[index] = None
    for index in with_d:
        match = re.search(error_pattern, index)
        i = match.group(1)
        if i in indicies: # found error index
            property_has_error[i] = index
        elif index not in property_has_error.keys(): # found new index
            property_has_error[index] = False
    return property_has_error

def _is_missing_scalar(value) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False

def _df_to_Dataset_arr(df: pd.DataFrame):
    arr = []

    property_has_error = _get_column_with_error_indicies(df.columns)

    for i, row in df.iterrows():
        dataset = Dataset()
        for index, error_index in property_has_error.items():
            value = row[index]
            error = row[error_index] if error_index else None;

            if isinstance(value, pd.Series):
                print("Warning! Multiple Columns have the same name. Taking first value")
                value = value.iat[0]
            if isinstance(error, pd.Series):
                print("Warning! Multiple Columns have the same name. Taking first value")
                error = error.iat[0]

            if _is_missing_scalar(error):
                error = None
            else:
                try:
                    error = float(error)
                    if not np.isfinite(error) or error == 0:
                        error = None
                except Exception:
                    pass

            if _is_missing_scalar(value):
                value = None
            else:
                try:
                    value = float(value)
                    if not np.isfinite(value):
                        value = None
                except Exception:
                    pass

            dataset[index] = Measurement(value, error) if error else value
        arr.append(dataset)

    return arr;

class DataCluster:
    @staticmethod
    def load_csv(filename: str, section: str | None = None) -> 'DataCluster':
        from ..tables.csv_table import load_csv
        return DataCluster(load_csv(filename, section))

    # ==================================================

    @staticmethod
    def _normalize_datasets(
        datasets: Optional[Union[Sequence[Dataset | Mapping], np.ndarray, pd.DataFrame]]
    ) -> list[Dataset]:
        if datasets is None:
            return []

        if isinstance(datasets, pd.DataFrame):
            return _df_to_Dataset_arr(datasets)

        if isinstance(datasets, np.ndarray):
            datasets = datasets.tolist()

        normalized = []
        for item in datasets:
            if isinstance(item, Dataset):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(Dataset(item))
            else:
                raise ValueError("Data Objects should manage only datasets or mappings!")

        return normalized

    def __init__(self, datasets: Optional[Union[Sequence[Dataset | Mapping], np.ndarray, pd.DataFrame]] = None):
        from batfloman_praktikum_lib.io.table_metadata import TableMetadataManager

        self.data = self._normalize_datasets(datasets)
        self.metadata_manager = TableMetadataManager()

    # ==================================================

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.column(index)
        if isinstance(index, slice):
            return DataCluster(self.data[index])
        return self.data[index]

    def __setitem__(self, index, value):
        if isinstance(index, str):
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                if len(self.data) == 0:
                    raise IndexError("Cannot assign a scalar column on an empty DataCluster")
                values = [value] * len(self.data)
            else:
                values = list(value)
                if len(self.data) == 0:
                    self.data = [Dataset({index: item}) for item in values]
                    return
                if len(values) != len(self.data):
                    raise ValueError("Column length must match the number of datasets")

            for dataset, item in zip(self.data, values):
                dataset[index] = item
            return

        if isinstance(index, slice):
            replacement = self._normalize_datasets(value)
            self.data[index] = replacement
            return

        if isinstance(value, Dataset):
            self.data[index] = value
        elif isinstance(value, Mapping):
            self.data[index] = Dataset(value)
        else:
            raise TypeError("Expected Dataset or mapping")

    def __iter__(self):
        return iter(self.data)

    # ==================================================
    
    def add(self,
        to_add: Dataset | dict | Iterable[Dataset | dict]
    ) -> None:
        if isinstance(to_add, (Dataset, dict)):
            add_arr = [to_add]
        elif isinstance(to_add, Iterable):
            add_arr = list(to_add)
        else:
            raise TypeError("Invalid input")

        normalized = []
        for item in add_arr:
            if isinstance(item, Dataset):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(Dataset(item))
            else:
                raise TypeError("Expected Dataset or dict")

        self.data.extend(normalized)

    def remove(self, to_remove: Dataset | list[Dataset]) -> None:
        if not isinstance(to_remove, list):
            to_remove = [to_remove]
        self.data = [dataset for dataset in self.data if dataset not in to_remove]

    def get_column_names(self) -> List[str]:
        columns = []
        for dataset in self.data:
            for key in dataset.measurements.keys():
                if key not in columns:
                    columns.append(key)
        return columns

    # ==================================================

    def sort(self, *keys: str) -> "DataCluster":
        if len(keys) == 0:
            raise ValueError("At least one sort key must be provided")

        sort_keys = list(keys)
        if not all(isinstance(key, str) for key in sort_keys):
            raise TypeError("Sort keys must be strings")

        def normalize_value(dataset: Dataset, index: str):
            has_index = index in dataset
            if not has_index:
                return (has_index, None)

            value = dataset[index]
            if isinstance(value, Measurement):
                return (has_index, value.value)

            try:
                return (has_index, float(value))
            except (ValueError, TypeError):
                return (has_index, value)

        def sort_key(dataset: Dataset):
            return tuple(normalize_value(dataset, index) for index in sort_keys)

        self.data = sorted(self, key=sort_key)

        return self
    
    def filter(self, condition: Callable[['Dataset'], bool]) -> 'DataCluster':
        filtered = [copy.deepcopy(ds) for ds in self.data if condition(ds)]
        return DataCluster(filtered)

    # ==================================================

    def column(self, index: str) -> np.ndarray:
        if index not in self.get_column_names():
            raise IndexError()
        
        return np.array(
            [(dataset[index] if (index in dataset) else "-") for dataset in self]
        )

    def values(self, index: str) -> np.ndarray:
        if index not in self.get_column_names():
            raise IndexError()

        def get_value(x):
            if isinstance(x, Measurement):
                return x.value
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan
    
        return np.array(
            [(get_value(dataset[index]) if (index in dataset) else "-") for dataset in self]
        )
        
    def errors(self, index: str) -> np.ndarray:
        if index not in self.get_column_names():
            raise IndexError()
        
        def get_error(x):
            if isinstance(x, Measurement):
                return x.error
            else:
                return 0
    
        return np.array(
            [(get_error(dataset[index]) if (index in dataset) else "-") for dataset in self]
        )

    # ==================================================

    def to_numpy(self, use_indicies = None, exclude_indicies = None, with_header=True) -> np.ndarray:
        indicies = self.get_column_names() if use_indicies is None else use_indicies;
        if exclude_indicies is not None:
            indicies = [i for i in indicies if i not in exclude_indicies];

        arr = [
            [dataset[i] if (i in dataset) else "-" for i in indicies] 
            for dataset in self
        ]

        return np.vstack([indicies, arr]) if with_header else np.array(arr)
    
    def to_dataframe(self, use_indicies = None, exclude_indicies = None) -> pd.DataFrame:
        indicies = self.get_column_names() if use_indicies is None else use_indicies;
        if exclude_indicies is not None:
            indicies = [i for i in indicies if i not in exclude_indicies];

        arr = self.to_numpy(use_indicies=use_indicies, exclude_indicies=exclude_indicies, with_header=False)
        df = pd.DataFrame(arr, columns=indicies)
        return df;

    def mean(self) -> dict:
        means = {}

        for index in self.get_column_names():
            values = self.values(index)
            errors = self.errors(index)

            # Filter out NaN values and their corresponding errors
            valid_indices = ~np.isnan(values)
            values = values[valid_indices]
            errors = errors[valid_indices]

            mean = np.mean(values)
            error = np.sqrt(np.sum(errors**2)) / len(errors)

            means[index] = Measurement(mean, error)
        
        return means

    
    # ==================================================

    def _latex_format_data(self, use_indicies=None, exclude_indicies=None):
        """
            This method formats the header & columns
        """
        # filter indicies
        indicies = self.get_column_names() if use_indicies is None else use_indicies;
        if exclude_indicies is not None:
            indicies = [i for i in indicies if i not in exclude_indicies];

        # format
        header = []
        columns = []

        for i in indicies:
            metadata = self.metadata_manager.get_metadata(i)
            header.append(latex_formatter.format_header(metadata, i));
            columns.append(self._format_column_data(i))
        data = np.column_stack(columns) # stack side by side -> 2d array

        return np.vstack([header, data]) # plop header on top

    def _format_column_data(self, index):
        column_data = self.column(index)
        metadata = self.metadata_manager.get_metadata(index);

        return latex_formatter.format_column_data(column_data, metadata)

    # ==================================================

    # def save_excel(self, filename: str) -> None:
    #     raise NotImplementedError("save to excel not implemented!")
    #
    # def save_csv(self, filename: str) -> None:
    #     raise NotImplementedError("save to csv not implemented!")
    #
    # def save_latex(self, filename: str, 
    #     *, 
    #     print_success_msg: bool = True,
    #     auto_create_dirs: bool = False,
    #     use_indices=None, 
    #     exclude_indices=None
    # ) -> None:
    #     from ..io.latex import save_latex
    #
    #     save_latex(self, filename,
    #         print_success_msg=print_success_msg,
    #         auto_create_dirs=auto_create_dirs,
    #         tableMetadata=self.metadata_manager,
    #         use_indices=use_indices,
    #         exclude_indices=exclude_indices,
    #     )
    #     # df = self.to_dataframe(use_indicies=use_indices, exclude_indicies=exclude_indices)
    #     # tables.export_as_latex_table(df, filename, metadata_manager=self.metadata_manager)
    
    # ==================================================
    # json

    # def to_json(self, indent: Optional[int] = None) -> str:
    #     data = [json.loads(ds.to_json()) for ds in self.data]
    #     return json.dumps(data, indent=indent)
    #
    # @staticmethod
    # def from_json(json_str: str):
    #     raw_list = json.loads(json_str)
    #     datasets = [Dataset.from_json(json.dumps(d)) for d in raw_list]
    #     return DataCluster(datasets)
    #
    # def save_json(self, path: str, indent: Optional[int] = 3):
    #     path = ensure_extension(path, ".json")
    #
    #     with open(path, "w", encoding="utf-8") as f:
    #         f.write(self.to_json(indent=indent))
    #
    # @staticmethod
    # def load_json(path: str) -> "DataCluster":
    #     path = ensure_extension(path, ".json")
    #
    #     with open(path, "r", encoding="utf-8") as f:
    #         json_str = f.read()
    #     return DataCluster.from_json(json_str)

    # ==================================================

    def __str__(self):
        columns = self.get_column_names()

        # Prepare rows as lists of strings exactly how they will be printed
        rows = []
        for row in self:
            arr = [(row[i] if i in row else "-") for i in columns]

            def is_empty_or_none(value):
                return value is None or (isinstance(value, str) and value.strip() == "")

            def to_str(value) -> str:
                try:
                    if isinstance(value, Measurement):
                        return f"{value}"
                    return str(value)
                except Exception:
                    return repr(value)


            arr = [("-" if is_empty_or_none(x) else to_str(x)) for x in arr]
            rows.append(arr)

        # Compute column widths based on both headers and formatted rows
        column_widths = [
            max(len(columns[i]), *(len(r[i]) for r in rows))
            for i in range(len(columns))
        ]

        # Build table string
        table = " | ".join(f"{col:{w}}" for col, w in zip(columns, column_widths)) + "\n"
        table += "-+-".join("-" * w for w in column_widths) + "\n"

        for arr in rows:
            table += " | ".join(f"{val:<{w}}" for val, w in zip(arr, column_widths)) + "\n"

        return table

    def print(self):
        print(self.__str__())

    def to_dict(self) -> dict:
        return {
            "__type__": "DataCluster",
            "data": [dataset.to_dict() for dataset in self.data],
            "metadata": {
                index: asdict(metadata)
                for index, metadata in self.metadata_manager._metadata.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataCluster":
        datasets = [Dataset.from_dict(dataset) for dataset in data["data"]]
        cluster = cls(datasets)

        for index, metadata in data.get("metadata", {}).items():
            cluster.metadata_manager.set_metadata(index, metadata)

        return cluster

    def to_json(self, *, indent: int | None = 2) -> str:
        from batfloman_praktikum_lib.io.json import dumps_json
        return dumps_json(self, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DataCluster":
        from batfloman_praktikum_lib.io.json import loads_json
        return loads_json(json_str)

    def save_json(self, path: str, *, indent: int | None = 2) -> str:
        from batfloman_praktikum_lib.io.json import save_json
        return save_json(self, path, indent=indent)

    @classmethod
    def load_json(cls, path: str) -> "DataCluster":
        from batfloman_praktikum_lib.io.json import load_json
        return load_json(path)
