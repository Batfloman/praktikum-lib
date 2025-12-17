from batfloman_praktikum_lib import graph_fit, rel_path, graph

import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use-cache", action="store_true")
args = parser.parse_args()

# ==================================================

def gaussian(x, A, sigma, x0):
    return A * np.exp(- ( (x-x0)/sigma )**2 )

def model(x, A1, s1, x1, A2, s2, x2, m, n):
    return gaussian(x, A1, s1, x1) + gaussian(x, A2, s2, x2) + graph_fit.Linear.model(x, m, n)

# x-Werte zufällig erzeugen
x_data = np.random.uniform(0, 9, 350)  # 50 zufällige Punkte zwischen -5 und 5
y_data = model(x_data, 
    2e-3, 0.5, 3,
    4e-3, 1, 5,
    0.001, 1
)
# add noise
noise = np.random.normal(0, 1e-4, size=y_data.shape)
sigma = np.full_like(y_data, 1e-4)  # entspricht dem Noise-Std
y_data = y_data + noise


initial_guess = graph_fit.find_init_params(
    model, 
    x_data, 
    y_data, 
    cachePath=rel_path("./fitcache.json", __file__),
    use_cache = args.use_cache,
)

res = graph_fit.least_squares_fit(model, x_data, y_data, y_err=sigma, initial_guess=initial_guess, param_names=["A1, s1, x1, A2, s2, x2, m, n"])

print(res.params)

# ==================================================
# plot

plot = graph.create_plot()
fig, ax = plot

graph.scatter(x_data, y_data, plot=plot, zorder=2)
graph.plot_func(res.func_no_err, plot=plot, label=fr"$\chi^2_\mathrm{{red}}$ = {res.quality}", color="red")

ax.legend()
plt.show()
