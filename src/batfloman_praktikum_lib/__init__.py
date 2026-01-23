from . import structs
from . import util
from . import graph_old, graph_fit
from . import function_analysis
from . import tables
from . import io

from .structs import DataCluster, Dataset, Measurement
# from .saving import save_latex
from .io import save_latex
from .path_managment import set_file, rel_path, ensure_extension, validate_filename
from .flags import check_quiet

from . import graph

__all__ = [
    # modules
    "graph",
    "graph_fit",
    "tables",
    "io",
    # files 
    "graph_old",
    "util",
    "function_analysis",

    "DataCluster",
    "Dataset",
    "Measurement",

    # helper
    "set_file",
    "rel_path",
    "save_latex",
    "ensure_extension",
    "validate_filename",
    "check_quiet",
]

def lazy_imports():
    global significant_rounding
    import batfloman_praktikum_lib.significant_rounding as significant_rounding

lazy_imports()
