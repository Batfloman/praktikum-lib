import numpy as np
import pandas as pd
import re
import copy
from typing import List, Union, Callable, Optional

from batfloman_praktikum_lib.tables.latex_table import formatter as latex_formatter
from batfloman_praktikum_lib.tables.metadata import MetadataManager
from batfloman_praktikum_lib.structs.measurement import Measurement
from batfloman_praktikum_lib.structs.dataset import Dataset 

from .. import tables

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

            try:
                error = float(error)
                if error == 0:
                    error = None
            except:
                pass
            try:
                value = float(value)
            except:
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

    def __init__(self, datasets: Optional[Union[List, np.ndarray, pd.DataFrame]] = None):
        if datasets is not None:
            if isinstance(datasets, pd.DataFrame):
                datasets = _df_to_Dataset_arr(datasets)  # keep your existing conversion
            elif isinstance(datasets, np.ndarray):
                datasets = datasets.tolist()  # convert NumPy array to list
            if not all(isinstance(obj, Dataset) for obj in datasets):
                raise ValueError("Data Objects should manage only datasets!")

        self.data = datasets or []
        self.metadata_manager = MetadataManager()

    # ==================================================

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data)[index]

    def __iter__(self):
        return iter(self.data)

    # ==================================================
    
    def add(self, to_add: Dataset | list[Dataset]) -> None:
        if not isinstance(to_add, list):
            to_add = [to_add]
        self.data.extend(to_add)

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

    def sort(self, key) -> None:
        def sort_key(x):
            has_index = key in x;
            if not has_index:
                return (has_index, None)
            else:
                value = x[key]
                if isinstance(value, Measurement):
                    return (has_index, value.value)
                else:
                    try:
                        float_ = float(value)
                        return (has_index, float_)
                    except ValueError:
                        return (has_index, value)
            
        self.data = sorted(self, key=sort_key)
    
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

    def save_excel(self, filename: str) -> None:
        raise NotImplementedError("save to excel not implemented!")

    def save_csv(self, filename: str) -> None:
        raise NotImplementedError("save to csv not implemented!")

    def save_latex(self, filename: str, use_indices=None, exclude_indicies=None) -> None:
        df = self.to_dataframe(use_indicies=use_indices, exclude_indicies=exclude_indicies)
        tables.export_as_latex_table(df, filename, metadata_manager=self.metadata_manager)
    
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
                if isinstance(value, Measurement):
                    return f"{value}";
                return str(value)


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
