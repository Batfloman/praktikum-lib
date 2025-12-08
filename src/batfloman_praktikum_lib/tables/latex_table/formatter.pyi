from ..metadata import MetadataManager, ColumnMetadata
from typing import Optional

def format_header(index: str, metadata: ColumnMetadata) -> str:
    """
    Generate a LaTeX-ready header string for a column.

    Parameters
    ----------
    metadata : ColumnMetadata
        Metadata object containing optional name, unit, exponent, SI prefix flag.
    index : str
        Fallback column name if metadata.name is not specified.

    Returns
    -------
    str
        Formatted header string, e.g. 'Voltage in V', or 'Energy in J x 10^3'.
    """
    ...

def get_single_column_format(
    index: str, 
    metadata: Optional[ColumnMetadata]
) -> str:
    """
    Generates the LaTeX `column_format` for a single index

    metadata (if `metadata_manager` is provided) and determines:
        - the alignment ('l', 'c', 'r')
        - optional left and right vertical borders ('|')

    If no `metadata` is given, default to `DEFAULT_ALIGNMENT` ('c').

    Parameters
    ----------
    index : str
        column identifier
    metadata : Optional[MetadataManager]
        An optional metadata Object that contains formatting info for the `index`-column.

    Returns
    -------
    str
        A single string representing the LaTeX column format for the given `index`.
        Example: 'c|' → centered with a right border

    """
    ...

def get_column_format(
    indices = list[str],
    metadata_manager: Optional[MetadataManager] = None
) -> str:
    """
    Generate a LaTeX `column_format` string for a table based on column metadata.

    For each column index in `indices`, this function checks the corresponding
    metadata (if `metadata_manager` is provided) and determines:
        - the alignment ('l', 'c', 'r')
        - optional left and right vertical borders ('|')
    
    If no `metadata_manager` is given, all columns default to `DEFAULT_ALIGNMENT` ('c').

    Parameters
    ----------
    indices : list[str]
        A list of column identifiers (matching the DataFrame columns or indices).
    metadata_manager : Optional[MetadataManager]
        An optional metadata manager that contains formatting info for each column.

    Returns
    -------
    str
        A single string representing the LaTeX column format.
        Example: 'c|l|l||r' → first column centered, second and third left-aligned
        with borders, last column right-aligned.

    Notes
    -----
    - If the alignment in the metadata is invalid, a warning is printed
      and the default alignment is used.
    - Borders are included according to `left_border` and `right_border` flags
      in the column metadata.
    """
    ...
