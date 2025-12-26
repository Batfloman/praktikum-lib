from batfloman_praktikum_lib import DataCluster, Dataset
from batfloman_praktikum_lib.io import save_latex

def test_plain():
    data = DataCluster()
    data.add({
        "a": 0.5,
        "b": 0.6,
    })
    save_latex(data)

