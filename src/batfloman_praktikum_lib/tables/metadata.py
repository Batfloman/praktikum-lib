from dataclasses import dataclass
from typing import Optional, Literal, Union, TypedDict
from ..significant_rounding.formatter import UncertaintyNotation

ALIGNMENT_VALUES = ["l", "c", "r"]   # nur zur Laufzeit
type ALIGNMENT = Literal["l", "c", "r"]
DEFAULT_ALIGNMENT: ALIGNMENT = "c"

@dataclass
class ColumnMetadata:
    """Holds formatting metadata for a column."""
    # header
    name: Optional[str] = None
    unit: Optional[str] = None
    display_exponent: Optional[int] = None
    use_si_prefix: Optional[bool] = True

    # values
    format_spec: Optional[str] = None

    # table format
    alignment: Literal["l", "c", "r"] = DEFAULT_ALIGNMENT
    left_border: bool = False
    right_border: bool = False

class ColumnMetadataDict(TypedDict, total=False):
    # Header
    name: str
    unit: str
    display_exponent: int
    use_si_prefix: bool

    # values
    format_spec: str

    # Table format
    alignment: ALIGNMENT  # "l", "c", "r"
    left_border: bool
    right_border: bool

DEFAULT_COLUMN_METADATA: ColumnMetadataDict = {
    "name": None,
    "unit": None,
    "display_exponent": None,
    "use_si_prefix": True,

    "format_spec": None,

    "alignment": DEFAULT_ALIGNMENT,
    "left_border": False,
    "right_border": False
}

def _md_from_deprecated(
        name: str | None = None,           # deprecated
        unit: str | None = None,           # deprecated
        display_exponent: int | None = None,  # deprecated
        force_exponent: bool | None = None,   # deprecated
        use_si_prefix: bool | None = None     # deprecated
) -> ColumnMetadataDict:
        # Merge deprecated parameters into dict/dataclass
        md_dict: ColumnMetadataDict = {}
        if name:
            md_dict["name"] = name 
        if unit:
            md_dict["unit"] = unit
        if display_exponent:
            md_dict["display_exponent"] = display_exponent
        if force_exponent:
            pass
        if use_si_prefix:
            md_dict["use_si_prefix"] = use_si_prefix

        return md_dict


class MetadataManager:
    def __init__(self):
        self._metadata: dict[str, ColumnMetadata] = {}

    def set_metadata(self, 
        index: str,
        md: Union[ColumnMetadata, ColumnMetadataDict] = DEFAULT_COLUMN_METADATA,
        *,
        name: str | None = None,           # deprecated
        unit: str | None = None,           # deprecated
        display_exponent: int | None = None,  # deprecated
        force_exponent: bool | None = None,   # deprecated
        use_si_prefix: bool | None = None     # deprecated
    ):
        """
        Set metadata for a column.

        - md: ColumnMetadata object or dict
        - Deprecated parameters: name, unit, display_exponent, force_exponent, use_si_prefix
        """
        # Check if deprecated parameters are used
        deprecated_used = any(param is not None for param in [name, unit, display_exponent, force_exponent, use_si_prefix])
        depr_md = {}
        if deprecated_used:
            print("*Warning*: Passing individual parameters (name, unit, ...) is deprecated. \nUse a dict or ColumnMetadata object instead.")
            depr_md = _md_from_deprecated(
                name = name,
                unit = unit,
                display_exponent=display_exponent,
                force_exponent=force_exponent,
                use_si_prefix=use_si_prefix
            )

        # Convert dict to ColumnMetadata if needed
        if isinstance(md, dict):
            md = {**md, **depr_md}
            md = ColumnMetadata(**md)
        elif not isinstance(md, ColumnMetadata):
            raise TypeError("md must be ColumnMetadata or dict")

        self._metadata[index] = md

    def get_metadata(self, index: str) -> ColumnMetadata:
        return self._metadata.get(index, None) or ColumnMetadata();

    def get_field(self, index: str, field: str):
        md = self.get_metadata(index)
        return getattr(md, field) if md else None

