from dataclasses import asdict
from batfloman_praktikum_lib.graph_fit.find_initial_parameters import InitialParameter, SliderSettings, order_initial_params

import pytest
import random

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

def test_initialParameters():
    x = InitialParameter(value = 5)
    assert asdict(x) == {"value": 5, "name": None}

    x = InitialParameter(value = 5e19)
    assert asdict(x) == {"value": 5e19, "name": None}

    x = InitialParameter(value = 5, name="x0")
    assert asdict(x) == {"value": 5, "name": "x0"}

    x = InitialParameter(value = 5, name="x0")
    assert InitialParameter.from_any(x) == x
    assert InitialParameter.from_any(asdict(x)) == x

    y = InitialParameter.from_any(x)
    assert x == y

    z = InitialParameter.from_any(asdict(x))
    assert x == z

    x = SliderSettings(
        slider_value=1.0,
        name="Test"
    )
    x = InitialParameter.from_any(x)
    assert asdict(x) == {"value": 1, "name": "Test"}

    x = SliderSettings(
        slider_value=1.0,
        exponent=1,
        name="Test"
    )
    x = InitialParameter.from_any(x)
    assert asdict(x) == {"value": 1e1, "name": "Test"}


def test_sliderSettings():
    for x in range(1000):
        slider_val = random.random() * 1000 - 500
        exponent = random.randint(-50, 50)
        x = SliderSettings(
            slider_value=slider_val,
            exponent = exponent,
        )
        assert asdict(x) == {"name": None, "slider_value": slider_val, "exponent": exponent};
        assert SliderSettings.from_any(x) == x
        assert SliderSettings.from_any(asdict(x)) == x

    for x in range(1000):
        slider_val = random.random() * 1000 - 500
        exponent = random.randint(-50, 50)
        x = SliderSettings(
            name = "test",
            slider_value=slider_val,
            exponent = exponent,
        )
        assert asdict(x) == {"name": "test", "slider_value": slider_val, "exponent": exponent};
        assert SliderSettings.from_any(x) == x
        assert SliderSettings.from_any(asdict(x)) == x


    x = SliderSettings.from_any({"name": "A", "value": 5e9}) 
    assert x == SliderSettings(name="A", slider_value = 5, exponent = 9)
