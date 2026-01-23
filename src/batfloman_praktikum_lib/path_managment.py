import os
from typing import Optional

_file = None;

def set_file(caller_file: str) -> None:
    global _file 
    _file = caller_file;

def rel_path(path: str, caller_file: Optional[str] = None) -> str:
    global _file

    # Fallback auf globalen Zustand
    if caller_file is None:
        if _file is None:
            raise ValueError("Need to set the file once!")
        caller_file = _file

    # Kein automatisches Überschreiben des globalen Zustands
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(caller_file)),
        path,
    ))

def ensure_extension(filename: str, extension: str) -> str:
    ext = extension.lower()
    if not filename.lower().endswith(ext):
        filename += extension
    return filename

def validate_filename(filename: str, extension: str) -> str:
    filename = ensure_extension(filename, extension)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    return filename

def create_dirs(filepath: str) -> None:
    dir = os.path.dirname(filepath)
    os.makedirs(dir, exist_ok=True)

def dir_exist(filepath: str) -> bool:
    dir = os.path.dirname(filepath)
    return os.path.exists(dir)
