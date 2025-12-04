from batfloman_praktikum_lib.graph_fit.find_initial_parameters import order_initial_params

import pytest

def test_order_initial_paramerters():
    def model1(x, A):
        return A
    assert order_initial_params(model1, [1]) == [1] 

    def model2(x, m, n):
        return m * x + n
    with pytest.raises(ValueError):
        order_initial_params(model2, [1])

    assert order_initial_params(model2, [1, 0.5]) == [1, 0.5]

    def model3(x, m, n):
        return m * x + n
    assert order_initial_params(model3, { "n": 0.5, "m": 1.0, "x": 1.0, }) == [1, 0.5]
    with pytest.raises(ValueError):
        order_initial_params(model3, { "n": 0.5, "x": 1.0, "y": 2 })

