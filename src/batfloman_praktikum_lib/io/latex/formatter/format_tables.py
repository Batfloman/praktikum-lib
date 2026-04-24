from typing import Optional

import numpy as np
import pandas as pd
import re

from batfloman_praktikum_lib.io.table_metadata import TableColumnMetadata, normalize_metadata
from batfloman_praktikum_lib.structs.measurement import Measurement

from ..optionTypes import TableOptions, normalize_table_options
from ._dataframe_helper import (
    get_column_format,
    normalize_metadata_manager,
    resolve_indices,
)
from ._number_helper import format_unit_latex
from .format_values import format_value


def _format_symbol(name: str) -> str:
    pattern = r"^[A-Za-z](?:([_^](?:\{[^{}]+\}|[A-Za-z0-9]+)))*"
    if re.match(pattern, name):
        return f"${name}$"
    return name


def format_table_header(
    index: str,
    metadata: TableColumnMetadata,
    *,
    sep: str = " in ",
) -> str:
    metadata = normalize_metadata(metadata)
    name = metadata.name or _format_symbol(index)

    unit_text = format_unit_latex(
        metadata.unit,
        exponent=metadata.display_exponent,
        use_si_prefix=metadata.use_si_prefix,
    )

    if not unit_text:
        return name

    return f"{name}{sep}{unit_text}"


def format_table_value(value, metadata: TableColumnMetadata) -> str:
    metadata = normalize_metadata(metadata)

    if isinstance(value, (str, np.str_)):
        try:
            value = float(value)
        except ValueError:
            return str(value)

    if isinstance(value, float) and np.isnan(value):
        return "NaN"

    display_exponent = metadata.display_exponent if hasattr(metadata, "display_exponent") else 0
    if display_exponent:
        value = value / 10**display_exponent

    format_spec = metadata.format_spec or ""
    if metadata.enforce_display_exponent and isinstance(value, Measurement) and "e" not in format_spec:
        format_spec = "f"

    return format_value(
        value,
        options={
            "format_spec": format_spec,
            "with_error": True,
        },
    )


def render_latex(
    column_format: str,
    formatted_headers: list[str],
    formatted_rows: list[list[str]],
) -> str:
    latex_str = ""
    latex_str += r"\begin{tabular}{" + column_format + "}" + "\n"
    latex_str += "\t" + r"\toprule" + "\n"
    latex_str += "\t" + (" & ".join(formatted_headers)) + r"\\" + "\n"
    latex_str += "\t" + r"\midrule" + "\n"

    for row in formatted_rows:
        latex_str += "\t" + (" & ".join(row)) + r"\\" + "\n"

    latex_str += "\t" + r"\bottomrule" + "\n"
    latex_str += r"\end{tabular}" + "\n"
    return latex_str


def format_dataframe(
    df: pd.DataFrame,
    *,
    options: Optional[TableOptions] = None,
) -> str:
    options = normalize_table_options(options)
    metadata = normalize_metadata_manager(options["metadata"])
    indices = resolve_indices(df, options)

    column_format = get_column_format(indices, metadata)
    formatted_headers = [
        format_table_header(
            col,
            metadata.get_metadata(col),
            sep=options["unit_separator"],
        )
        for col in indices
    ]

    formatted_rows = []
    for _, row in df.iterrows():
        formatted_rows.append([
            format_table_value(value, metadata.get_metadata(index))
            for index, value in zip(indices, row[indices])
        ])

    return render_latex(column_format, formatted_headers, formatted_rows)
