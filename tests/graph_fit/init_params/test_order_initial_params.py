import pytest
from types import SimpleNamespace
from batfloman_praktikum_lib.graph_fit.init_params.order_init_params import order_initial_params
from batfloman_praktikum_lib.graph_fit.init_params.parameterSlider import ParameterSlider

from batfloman_praktikum_lib.graph_fit import Gaussian

# --- Dummy Models ---

def linear(x, a, b):
    return a * x + b

# --- Tests ---

def test_sequence_input():
    seq = [2.0, 3.0]
    result = order_initial_params(linear, seq)
    assert result == seq

def test_mapping_input_with_floats():
    params = {"a": 2.0, "b": 3.0}
    result = order_initial_params(linear, params)
    assert result == [2.0, 3.0]

    params = {"b": 2.0, "a": 3.0}
    result = order_initial_params(linear, params)
    assert result == [3.0, 2.0]

    params = {"b": 2.0, "a": 3.0, "c": 5.0}
    result = order_initial_params(linear, params)
    assert result == [3.0, 2.0]

def test_mapping_input_with_slider():
    slider_a = SimpleNamespace(slider_value=5.0)
    slider_b = SimpleNamespace(slider_value=7.0)
    params = {"a": slider_a, "b": slider_b}
    result = order_initial_params(linear, params)
    assert result == [5.0, 7.0]

    params = {"b": slider_b, "a": slider_a}
    result = order_initial_params(linear, params)
    assert result == [5.0, 7.0]

def test_mapping_input_with_slider_real():
    # Dummy Matplotlib Axes für Konstruktion
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    slider_a = ParameterSlider("a", ax, initial_value=5.0, update_callback=lambda: None)
    slider_b = ParameterSlider("b", ax, initial_value=7.0, update_callback=lambda: None)
    
    params = {"a": slider_a, "b": slider_b}
    
    def linear(x, a, b):
        return a*x + b
    
    result = order_initial_params(linear, params)
    assert result == [5.0, 7.0]

    plt.close(fig)

def test_mapping_missing_param():
    params = {"a": 2.0}
    with pytest.raises(ValueError):
        order_initial_params(linear, params)

def test_mapping_missing_param_sequence():
    params = [2.0]
    with pytest.raises(ValueError):
        order_initial_params(linear, params)

def test_wrong_type_input():
    with pytest.raises(TypeError):
        order_initial_params(linear, 42)  # Not sequence or mapping

def test_fitmodel_class_input():
    fmodel = Gaussian
    seq = [1.0, 2.0, 3.0]
    result = order_initial_params(fmodel, seq)
    assert result == seq

def test_fitmodel_input():
    fmodel = Gaussian()
    seq = [1.0, 2.0, 3.0]
    result = order_initial_params(fmodel, seq)
    assert result == seq
