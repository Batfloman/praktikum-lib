from batfloman_praktikum_lib import graph_fit, rel_path
from batfloman_praktikum_lib.graph_fit import FitModel, Gaussian, Linear, CompositeFitModel, find_init_params
from batfloman_praktikum_lib import graph
import numpy as np
import random

def test_a():
    comp = Gaussian + Gaussian + Linear
    comp2 = Linear + Linear + Linear
    print(comp)
    assert issubclass(comp, FitModel)
    assert issubclass(comp, CompositeFitModel)
    assert issubclass(comp2, FitModel)
    assert issubclass(comp2, CompositeFitModel)

    A = random.normalvariate(100, 5)
    s = random.normalvariate(10, 5)
    x0 = random.normalvariate(0, 1)

    A2 = random.normalvariate(69, 5)
    s2 = random.normalvariate(20, 5)
    x2 = random.normalvariate(15, 1.5)

    m = random.normalvariate(0.5, 0.1)
    n = random.normalvariate(0, 0.1)
    def test_model(x):
        return Gaussian.model(x, A, s, x0) + Gaussian.model(x, A2, s2, x2) + Linear.model(x, m, n)

    xmin, xmax = -20, 100

    x_noise = np.array([random.uniform(0.95, 1.05) for _ in range(200)])
    x_data = np.array([random.random()*(xmax - xmin) + xmin for _ in range(200)])
    x_data = x_data + x_noise

    y_noise = np.array([random.uniform(0.95, 1.05) for _ in range(200)])
    y_data = test_model(x_data) * y_noise

    params = find_init_params(
        comp,
        x_data, y_data,
        cachePath=rel_path("./chace", __file__)
    )

    fit = graph_fit.least_squares_fit(
        comp,
        x_data,
        y_data,
        initial_guess=params
    )
    print(fit.params)

    plot = graph.create_plot()
    graph.scatter(x_data, y_data, plot=plot)
    graph.plot_func(test_model, plot=plot, interval=(xmin, xmax))
    graph.plot_func(fit, plot=plot, interval=(xmin, xmax), color="red", zorder=4)
    graph.plt.show()

