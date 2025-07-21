import numpy as np
import sys
sys.path.append('../')
import praktikum_lib
from praktikum_lib.structs import Measurement

a = Measurement(3.0, "10%")
b = Measurement(2.0, 0.1)

arr = np.array([a, b])
print(arr**2)
