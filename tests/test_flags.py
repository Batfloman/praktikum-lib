import sys

import pytest

from batfloman_praktikum_lib import flags


@pytest.fixture(autouse=True)
def reset_flag_runtime_state():
    flags._reset_runtime_state_for_tests()
    yield
    flags._reset_runtime_state_for_tests()


def test_should_skip_popup_sequence_consumes_skip_count(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--skip-popups", "2"])

    assert flags.check_quiet() is False
    assert flags.should_skip_popup_sequence() is True
    assert flags.should_skip_popup_sequence() is True
    assert flags.should_skip_popup_sequence() is False


def test_should_skip_popup_sequence_honors_quiet(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--quiet", "--skip-popups", "2"])

    assert flags.check_quiet() is True
    assert flags.should_skip_popup_sequence() is True
    assert flags.should_skip_popup_sequence() is True
