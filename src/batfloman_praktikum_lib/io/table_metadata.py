from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union, TypedDict, NotRequired

type ALIGNMENT = Literal["l", "c", "r"]
ALIGNMENT_VALUES = ["l", "c", "r"]
DEFAULT_ALIGNMENT: ALIGNMENT = "c"

@dataclass
class TableColumnMetadataClass:
    # header
    name: Optional[str] = None
    unit: Optional[str] = None
    display_exponent: int = 0
    enforce_display_exponent: bool = False
    use_si_prefix: bool = True
    # values
    format_spec: Optional[str] = None
    # format
    alignment: ALIGNMENT = DEFAULT_ALIGNMENT
    left_border: bool = False
    right_border: bool = False

    @classmethod
    def from_any(cls, any):
        if isinstance(any, TableColumnMetadataClass):
            return any
        if isinstance(any, dict):
            return TableColumnMetadataClass(**any);
        raise ValueError("")

class TableColumnMetadataDict(TypedDict):
    # header
    name: NotRequired[str]
    unit: NotRequired[str]
    display_exponent: NotRequired[int]
    enforce_display_exponent: NotRequired[bool]
    use_si_prefix: NotRequired[bool]
    # values
    format_spec: NotRequired[str]
    # format
    alignment: NotRequired[ALIGNMENT]
    left_border: NotRequired[bool]
    right_border: NotRequired[bool]

type TableColumnMetadata = TableColumnMetadataClass | TableColumnMetadataDict

def normalize_metadata(md: TableColumnMetadata) -> TableColumnMetadataClass:
    if isinstance(md, TableColumnMetadataClass):
        return md
    if isinstance(md, dict):
        return TableColumnMetadataClass(**md)
    raise TypeError("md must be TableColumnMetadata or dict")

class TableMetadataManager:
    def __init__(self):
        self._metadata: dict[str, TableColumnMetadataClass] = {}

    def set_metadata(self,
        index: str,
        md: Optional[TableColumnMetadata] = None
    ):
        self._metadata[index] = normalize_metadata(md or {})

    def update_metadata(self, 
        index: str, 
        md: Optional[TableColumnMetadata] = None
    ):
        md = md or {}
        new = md if isinstance(md, dict) else asdict(md)
        old = {} if index not in self._metadata else asdict(self._metadata[index])
        combined = {**old, **new}
        self._metadata[index] = TableColumnMetadataClass(**combined)

    def get_metadata(self, index: str) -> TableColumnMetadataClass:
        return self._metadata.get(index, TableColumnMetadataClass())

    def get_field(self, index: str, field: str):
        return getattr(self.get_metadata(index), field, None)
