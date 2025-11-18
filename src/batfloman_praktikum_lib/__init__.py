from . import structs
from . import util
from . import graph_old, graph_fit
from . import function_analysis
from . import tables

from .structs import DataCluster, Dataset, Measurement

from . import graph

__all__ = [
    # modules
    "graph",
    "graph_fit",
    "tables",
    # files 
    "graph_old",
    "util",
    "function_analysis",

    "DataCluster",
    "Dataset",
    "Measurement"
]

def lazy_imports():
    global significant_rounding
    import batfloman_praktikum_lib.significant_rounding as significant_rounding

lazy_imports()

import os

def rel_path(path: str, caller_file: str) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(caller_file)), path);
