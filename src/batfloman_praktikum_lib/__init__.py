from . import structs
from . import util
from . import graph, graph_fit
from . import function_analysis
from . import tables

from .structs import DataCluster, Dataset, Measurement

__all__ = [
    # modules
    "graph_fit",
    "tables",
    # files 
    "graph",
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
