import os;

def ensure_extension(filename:str, extension: str) -> str:
    # Ensures the filename has the correct extension
    if not filename.endswith(extension):
        filename += extension;
    return filename;

def validate_filename(filename: str, extension: str) -> str:
    # Validates the existence of the file and appends the correct extension
    filename = ensure_extension(filename, extension)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    return filename;