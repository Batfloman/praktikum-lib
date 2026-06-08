import inspect
import os
from pathlib import Path
from typing import Optional

PathInput = str | os.PathLike[str]

_base_dir: Path | None = None

def set_file(caller_file: PathInput) -> None:
    global _base_dir
    _base_dir = Path(caller_file).expanduser().resolve().parent

def set_basedir(base_dir: PathInput, caller_file: Optional[PathInput] = None) -> None:
    global _base_dir
    path = Path(base_dir).expanduser()

    if not path.is_absolute():
        if caller_file is None:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                caller_file = frame.f_back.f_code.co_filename

        if caller_file is not None:
            path = Path(caller_file).expanduser().resolve().parent / path

    _base_dir = path.resolve()

def set_base_dir(base_dir: PathInput, caller_file: Optional[PathInput] = None) -> None:
    set_basedir(base_dir, caller_file)

def rel_path(path: PathInput, caller_file: Optional[PathInput] = None) -> Path:
    global _base_dir

    # Fallback auf globalen Zustand
    if caller_file is None:
        if _base_dir is None:
            raise ValueError("Need to set the file once!")
        base_dir = _base_dir
    else:
        base_dir = Path(caller_file).expanduser().resolve().parent

    return (base_dir / path).resolve()

def ensure_extension(filename: PathInput, extension: str) -> Path:
    ext = extension.lower()
    filename_str = os.fspath(filename)
    if filename_str.lower().endswith(ext):
        return Path(filename)
    return Path(filename_str + extension)

def validate_filename(filename: PathInput, extension: str) -> Path:
    filename = ensure_extension(filename, extension)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    return filename

def create_dirs(filepath: PathInput) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

def dir_exist(filepath: PathInput) -> bool:
    return Path(filepath).parent.exists()
