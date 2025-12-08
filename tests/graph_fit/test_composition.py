from batfloman_praktikum_lib import graph_fit
from batfloman_praktikum_lib.graph_fit import FitModel, Gaussian, Linear, CompositeFitModel
from batfloman_praktikum_lib import graph
import numpy as np
import random

def test_a():
    comp = Gaussian + Linear
    print(comp)
    assert issubclass(comp, FitModel)
    assert issubclass(comp, CompositeFitModel)

    A = random.normalvariate(100, 5)
    s = random.normalvariate(20, 5)
    x0 = random.normalvariate(5, 1)
    m = random.normalvariate(0.5, 0.1)
    n = random.normalvariate(0, 0.1)
    def test_model(x):
        return Gaussian.model(x, A, s, x0) + Linear.model(x, m, n)

    xmin, xmax = -20, 100

    x_noise = np.array([random.uniform(0.98, 1.02) for _ in range(200)])
    x_data = np.array([random.random()*(xmax - xmin) + xmin for _ in range(200)])
    x_data = x_data + x_noise

    y_noise = np.array([random.uniform(0.95, 1.05) for _ in range(200)])
    y_data = test_model(x_data) * y_noise

    fit = graph_fit.least_squares_fit(
        Gaussian + Linear,
        x_data,
        y_data
    )
    print(fit.params)

    plot = graph.create_plot()
    graph.scatter(x_data, y_data, plot=plot)
    graph.plot_func(test_model, plot=plot, interval=(xmin, xmax))
    graph.plot_func(fit, plot=plot, interval=(xmin, xmax), color="red", zorder=4)
    graph.plt.show()

