from dataclasses import dataclass
from typing import Optional

@dataclass
class ColumnMetadata:
    """Holds formatting metadata for a column."""
    name: Optional[str] = None
    unit: Optional[str] = None
    display_exponent: Optional[int] = None
    force_exponent: Optional[bool] = None
    use_si_prefix: Optional[bool] = True

class MetadataManager:
    def __init__(self):
        self._metadata: dict[str, ColumnMetadata] = {}

    def set_metadata(
        self,
        index: str,
        name: str | None = None,
        unit: str | None= None,
        display_exponent: int | None = None,
        force_exponent: bool | None= None,
        use_si_prefix: bool | None = None,
    ):
        self._metadata[index] = ColumnMetadata(
            name=name,
            unit=unit,
            display_exponent=display_exponent,
            force_exponent=force_exponent,
            use_si_prefix=use_si_prefix,
        )

    def get_metadata(self, index: str) -> ColumnMetadata:
        return self._metadata.get(index, None) or ColumnMetadata();

    def get_field(self, index: str, field: str):
        md = self.get_metadata(index)
        return getattr(md, field) if md else None

