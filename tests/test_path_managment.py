from pathlib import Path

from batfloman_praktikum_lib import path_managment


def test_rel_path_uses_caller_parent_and_returns_path(tmp_path):
    caller_file = tmp_path / "script.py"
    caller_file.touch()

    path = path_managment.rel_path("output/result.txt", caller_file)

    assert isinstance(path, Path)
    assert path == (tmp_path / "output" / "result.txt").resolve()


def test_set_file_stores_base_dir(tmp_path, monkeypatch):
    caller_file = tmp_path / "nested" / "script.py"
    caller_file.parent.mkdir()
    caller_file.touch()

    monkeypatch.setattr(path_managment, "_base_dir", path_managment._base_dir)
    path_managment.set_file(caller_file)

    assert path_managment.rel_path("data.csv") == (caller_file.parent / "data.csv").resolve()


def test_set_base_dir_stores_base_dir_directly(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    base_dir.mkdir()

    monkeypatch.setattr(path_managment, "_base_dir", path_managment._base_dir)
    path_managment.set_base_dir(base_dir)

    assert path_managment.rel_path("data.csv") == (base_dir / "data.csv").resolve()


def test_set_basedir_resolves_relative_to_calling_file(monkeypatch):
    monkeypatch.setattr(path_managment, "_base_dir", path_managment._base_dir)

    path_managment.set_basedir("./base")

    assert path_managment.rel_path("data.csv") == (Path(__file__).parent / "base" / "data.csv").resolve()


def test_set_base_dir_can_resolve_relative_to_explicit_caller(tmp_path, monkeypatch):
    caller_file = tmp_path / "main.py"
    caller_file.touch()

    monkeypatch.setattr(path_managment, "_base_dir", path_managment._base_dir)
    path_managment.set_base_dir("./base", caller_file)

    assert path_managment.rel_path("data.csv") == (tmp_path / "base" / "data.csv").resolve()


def test_ensure_extension_accepts_paths(tmp_path):
    assert path_managment.ensure_extension(tmp_path / "data", ".txt") == tmp_path / "data.txt"
    assert path_managment.ensure_extension(tmp_path / "data.TXT", ".txt") == tmp_path / "data.TXT"


def test_path_validation_accepts_paths(tmp_path):
    assert path_managment.ensure_extension(tmp_path / "table", ".xlsx") == tmp_path / "table.xlsx"
