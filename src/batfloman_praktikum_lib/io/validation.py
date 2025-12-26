import os

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
