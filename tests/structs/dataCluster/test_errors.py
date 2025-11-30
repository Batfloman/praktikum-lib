from batfloman_praktikum_lib import DataCluster
from batfloman_praktikum_lib.structs.measurement import Measurement
from ...generate_mock_data import generate_random_measurement, generate_random_dataset, MeasurementSetting

def test_x():
    dsets = [generate_random_dataset([
        MeasurementSetting(name ="x", mu=i*3), 
        MeasurementSetting(name ="y", mu=i*5), 
        i
    ]) for i in range(1, 10)]

    print("\n")
    data = DataCluster(dsets)
    print(data)
