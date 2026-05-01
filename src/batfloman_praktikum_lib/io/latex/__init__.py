# bind singledispatch
from .save import to_latex, save_latex
from .optionTypes import ValueOptions, TableOptions

__all__ = [
    "to_latex",
    "save_latex",
    "ValueOptions",
    "TableOptions",
]
