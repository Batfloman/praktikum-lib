from batfloman_praktikum_lib import graph_fit, rel_path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import inspect

import os
import json
from pathlib import Path

# ==================================================


def gaussian(x, A, sigma, x0):
    return A * np.exp(- ( (x-x0)/sigma )**2 )

def model(x, A1, s1, x1, A2, s2, x2, m, n):
    return gaussian(x, A1, s1, x1) + gaussian(x, A2, s2, x2) + graph_fit.Linear.model(x, m, n)

# x-Werte zufällig erzeugen
x_data = np.random.uniform(0, 9, 35)  # 50 zufällige Punkte zwischen -5 und 5
y_data = model(x_data, 
    2e-3, 0.5, 3,
    4e-3, 1, 5,
    0.001, 1
)

params = graph_fit.find_init_params(x_data, y_data, model, cachePath=rel_path("./fitcache.json", __file__), index=1)
print(params)

plt.show()
