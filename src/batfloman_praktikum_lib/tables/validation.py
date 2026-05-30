import os
from pathlib import Path

PathInput = str | os.PathLike[str]

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

