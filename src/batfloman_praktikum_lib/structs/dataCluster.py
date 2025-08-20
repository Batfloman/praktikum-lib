import numpy as np
import pandas as pd
import re

from typing import List, Union

from .dataset import Dataset
from .measurement import Measurement
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

    def __init__(self, datasets: Union[List[Measurement], pd.DataFrame] = None):
        if isinstance(datasets, pd.DataFrame):
            datasets = _df_to_Dataset_arr(datasets)

        self.data = datasets or [];
        self.column_metadata = {};

    # ==================================================

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dataset:
        return self.data[index]

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
    
    def filter() -> 'DataCluster':
        pass

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
    
    def to_dataframe(use_indicies = None, exclude_indicies = None) -> pd.DataFrame:
        df = pd.DataFrame()
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

    def set_column_metadata(self, index, name=None, display_exponent=None, force_exponent=None, unit=None):
        if index not in self.get_column_names():
            print(f"Warning: No Dataset has index {index}");

        if index not in self.column_metadata:
            self.column_metadata[index] = {};
        
        if name is not None:
            self.column_metadata[index]['name'] = name;
        if display_exponent is not None and isinstance(display_exponent, int):
            self.column_metadata[index]['display_exponent'] = display_exponent
        if force_exponent is not None and isinstance(force_exponent, int):
            self.column_metadata[index]['force_exponent'] = force_exponent
        if unit is not None and isinstance(unit, str):
            self.column_metadata[index]['unit'] = unit

    def _get_column_metadata(self, index, field):
        if not index in self.column_metadata:
            return;
        
        metadata = self.column_metadata[index]
        if not isinstance(metadata, dict):
            print(f"Metadata should be stored as an 'dict' not {type(metadata)}")
            return;
        
        return metadata.get(field)
    
    def _format_column_header(self, index):
        name = self._get_column_metadata(index, 'name') or index;
        unit = self._get_column_metadata(index, 'unit');
        exponent = self._get_column_metadata(index, 'display_exponent');

        if exponent is None or exponent == 0:
            exponent_text = "";
        else:
            # exponent_text = fr"$\times 10^{{{exponent}}}$ "
            exponent_text = fr"$10^{{{exponent}}}$ "

        if unit is None or unit == "":
            if exponent_text == "":
                unit_text = ""
            else :
                unit_text = f"[{exponent_text}]";
        else:
            unit_text = f"[{exponent_text}{unit}]";

        return f"{name} {unit_text}";

    def _format_column_data(self, index) -> np.ndarray:
        column_data = self.column(index)
        display_exponent = self._get_column_metadata(index, 'display_exponent')
        force_exponent = self._get_column_metadata(index, 'force_exponent')

        if force_exponent is not None and not isinstance(force_exponent, bool):
            print(f"Warning: 'force_exponent' should be a bool and not {type(force_exponent)}"); 

        if isinstance(display_exponent, int):
            def format_value(val):
                if str(val).strip() == "-":
                    return "-";

                num_val = float(val) if isinstance(val, str) else val
                shifed_val = num_val / 10**display_exponent

                if isinstance(shifed_val, Measurement):
                    return shifed_val if not force_exponent else shifed_val.to_str_bracket(0)
                
                return np.round(shifed_val, 10) # float uncertainty removed!
            return [format_value(val) for val in column_data];

            # format_shift_exponent = lambda x: x / 10**display_exponent;
            # formatted = [(format_shift_exponent(float(val) if isinstance(val, str) else val) if str(val).strip() != "-" else "-") for val in column_data]
            # return formatted
        elif display_exponent is not None:
            print(f"Display exponten was not of type 'int'. Was: {type(display_exponent)}")

        return column_data

    def create_display_array(self, use_indicies=None, exclude_indicies=None):
        indicies = self.get_column_names() if use_indicies is None else use_indicies;
        if exclude_indicies is not None:
            indicies = [i for i in indicies if i not in exclude_indicies];
        
        header = [self._format_column_header(i) for i in indicies]

        columns = []
        for i in indicies:
            columns.append(self._format_column_data(i))
        
        data = np.column_stack(columns)

        return np.vstack([header, data])


    # ==================================================

    def save_excel(self, filename: str) -> None:
        pass

    def save_csv(self, filename: str) -> None:
        pass

    def save_latex(self, filename: str, use_indices=None, exclude_indicies=None) -> None:
        data = self.create_display_array(use_indicies=use_indices, exclude_indicies=exclude_indicies)
        # data = self.to_numpy(use_indicies=use_indices, exclude_indicies=exclude_indicies, with_header=True)
        tables.export_as_latex(data, filename, has_header=True)
    
    # ==================================================


    def __str__(self):
        columns = self.get_column_names()

        header_widths = [len(col) for col in columns]
        data_widths = [max([len(str(val)) for val in self.column(column)]) for column in columns]
        column_widths = [max(x, y) for x, y in zip(header_widths, data_widths)]

        table = " | ".join(f"{column:{width}}" for column, width in zip(columns, column_widths)) + "\n"
        table += "-+-".join("-" * width for width in column_widths) + "\n"
        for row in self:
            arr = [(row[i] if i in row else "-") for i in columns]
            def is_empty_or_none(value):
                return value is None or (isinstance(value, str) and value.strip() == "")
            arr = [("-" if is_empty_or_none(x) else x) for x in arr]
            table += " | ".join(f"{value:<{width}}" for value, width in zip(arr, column_widths)) + "\n"

        return table;

    def print(self):
        print(self.__str__())
