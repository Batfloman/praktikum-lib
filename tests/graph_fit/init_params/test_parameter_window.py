from batfloman_praktikum_lib.graph_fit import Gaussian, Linear
from batfloman_praktikum_lib.graph_fit.init_params.parameterWindow import infer_parameter_groups


def test_infer_parameter_groups_for_repeated_model():
    model = 2 * Gaussian
    param_names = ["A_1", "s_1", "x0_1", "A_2", "s_2", "x0_2"]

    groups = infer_parameter_groups(param_names, model=model)

    assert groups is not None
    assert [group.title for group in groups] == ["Gaussian", "Gaussian"]
    assert groups[0].params == [("A_1", "A"), ("s_1", "s"), ("x0_1", "x0")]
    assert groups[1].params == [("A_2", "A"), ("s_2", "s"), ("x0_2", "x0")]


def test_infer_parameter_groups_for_mixed_composite_model():
    model = Gaussian + Linear
    param_names = ["A_1", "s_1", "x0_1", "m_2", "n_2"]

    groups = infer_parameter_groups(param_names, model=model)

    assert groups is not None
    assert [group.title for group in groups] == ["Gaussian", "Linear"]


def test_infer_parameter_groups_returns_none_for_non_composite_model():
    param_names = ["A", "s", "x0"]

    groups = infer_parameter_groups(param_names, model=Gaussian)

    assert groups is None
