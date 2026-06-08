import numbers
from functools import singledispatch
from typing import cast

import pandas as pd

from batfloman_praktikum_lib.path_managment import PathInput, create_dirs, dir_exist, ensure_extension, rel_path
from batfloman_praktikum_lib.io.termColors import bcolors
from batfloman_praktikum_lib.io.latex.optionTypes import TableOptions, ValueOptions

from .formatter import format_dataframe, format_value

@singledispatch
def to_latex(obj, options: ValueOptions | TableOptions) -> str:
    raise NotImplementedError(f"No LaTeX renderer for {type(obj)}")

# ==================================================
#  implemenations
# ==================================================

def register_impl(): 
    # to avoid circular imports, we import inside of a function
    from batfloman_praktikum_lib.structs.measurement import Measurement
    from batfloman_praktikum_lib.structs.dataCluster import DataCluster

    @to_latex.register
    def _(obj: numbers.Real, options: ValueOptions) -> str:
        return format_value(obj, None, options=options)

    @to_latex.register
    def _(obj: Measurement, options: ValueOptions) -> str:
        return format_value(obj.value, obj.error, options=options)

    @to_latex.register
    def _(obj: pd.DataFrame, options: TableOptions) -> str:
        return format_dataframe(obj, options=options)

    @to_latex.register
    def _(obj: DataCluster, options: TableOptions) -> str:
        return format_dataframe(obj.to_dataframe(), options=options)

register_impl()

# ==================================================
#  save
# ==================================================

def save_latex(
    obj,
    path: PathInput,
    options: ValueOptions | TableOptions | None = None,
    *,
    print_success_msg: bool = True,
    auto_create_dirs: bool = False,
) -> str:
    options = normalize_options(obj, options)
    latex_str = to_latex(obj, options)

    try:
        path = rel_path(path)
    except ValueError:
        pass

    path = ensure_extension(path, ".tex")
    if auto_create_dirs and not dir_exist(path):
        create_dirs(path)
        print(f"{bcolors.CREATED}Created directory: {bcolors.ENDC}{path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    if print_success_msg:
        print(f"{bcolors.OKGREEN}Succesfully saved{bcolors.ENDC}",
              f"{bcolors.OKBLUE}{type(obj).__name__}{bcolors.ENDC}")
        print(f"\t{bcolors.DIM}to {path}{bcolors.ENDC}")

    return latex_str


def normalize_options(obj, options: ValueOptions | TableOptions | None) -> ValueOptions | TableOptions:
    if options is None:
        if isinstance(obj, (pd.DataFrame,)):
            return {}
        return {}

    return cast(ValueOptions | TableOptions, options)
