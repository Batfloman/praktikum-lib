from dataclasses import dataclass, field
from typing import Optional, Iterable

from batfloman_praktikum_lib.io.table_metadata import TableColumnMetadata

@dataclass
class ValueOptions:
    format_spec: str = ""
    unit: Optional[str] = None
    use_si_prefix: bool = True
    fixed_exponent: Optional[int] = None
    with_error: bool = True

@dataclass
class TableOptions:
    metadata: Optional[dict[str, TableColumnMetadata]] = None
    use_indices: Optional[list[str]] = None
    exclude_indices: Optional[Iterable[str]] = None

@dataclass
class LatexOptions:
    value: ValueOptions = field(default_factory=ValueOptions)
    table: TableOptions = field(default_factory=TableOptions)
