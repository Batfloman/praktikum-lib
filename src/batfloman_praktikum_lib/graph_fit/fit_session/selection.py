from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from ...structs.dataCluster import DataCluster
from ...structs.dataset import Dataset
from .analysis import ComponentFitAnalysis, FitAnalysis, RecordConflictMode, _merge_record_data

if TYPE_CHECKING:
    from ..fitResult import FIT_METHODS, FitResult
    from .session import FitSession

type SelectionRef = str | int


class SelectionOptions(TypedDict):
    ref: SelectionRef
    component: NotRequired[SelectionRef | None]
    fields: NotRequired[Mapping[str, Any] | Dataset | None]
    rename: NotRequired[Mapping[str, str] | None]
    auto_fit: NotRequired[bool]
    method: NotRequired[FIT_METHODS | None]


type SelectionSpec = SelectionRef | SelectionOptions
type SelectionIterable = Iterable[SelectionSpec]
type MergeableSelection = FitSelection | FitSelectionCluster


def _normalize_fields(fields: Mapping[str, Any] | Dataset | None) -> Dataset:
    if fields is None:
        return Dataset()
    if isinstance(fields, Dataset):
        return fields.copy()
    return Dataset(fields)


def _apply_rename(
    record: Dataset,
    rename: Mapping[str, str] | None,
) -> Dataset:
    if rename is None:
        return record.copy()
    return record.rename(**dict(rename))


@dataclass
class FitSelection:
    session: "FitSession"
    ref: SelectionRef
    analysis: FitAnalysis
    component_ref: SelectionRef | None = None
    fields: Dataset = field(default_factory=Dataset)
    rename: Mapping[str, str] | None = None

    @property
    def fit_result(self):
        return self.analysis.fit_result

    @property
    def component(self) -> ComponentFitAnalysis | None:
        if self.component_ref is None:
            return None
        return self.analysis.component(self.component_ref)

    @property
    def params(self) -> Dataset:
        selected_component = self.component
        if selected_component is not None:
            return selected_component.params
        return self.analysis.params

    @property
    def extra(self) -> Dataset:
        return self.fields

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, value) -> None:
        self.fields[key] = value

    def to_record(
        self,
        *,
        fields: Mapping[str, Any] | Dataset | None = None,
        rename: Mapping[str, str] | None = None,
        on_conflict: RecordConflictMode = "raise",
    ) -> Dataset:
        selected_component = self.component
        merged_rename = dict(self.rename or {})
        if rename is not None:
            merged_rename.update(rename)

        if selected_component is not None:
            record = selected_component.to_record(
                fit_name=self.analysis.model_name,
                model_id=self.analysis.model_id,
            )
        else:
            record = self.analysis.to_record()

        resolved_record = _apply_rename(record, merged_rename or None)
        merged_fields = _merge_record_data(
            self.fields,
            _normalize_fields(fields),
            on_conflict=on_conflict,
        )
        return _merge_record_data(
            resolved_record,
            merged_fields,
            on_conflict=on_conflict,
        )

    def merge(self, *others: MergeableSelection) -> "FitSelectionCluster":
        return FitSelectionCluster([self]).merge(*others)


class FitSelectionFieldTable:
    def __init__(self, cluster: "FitSelectionCluster"):
        self.cluster = cluster

    def __len__(self) -> int:
        return len(self.cluster)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.cluster.to_datacluster()[key]
        if isinstance(key, slice):
            return DataCluster([
                selection.to_record()
                for selection in self.cluster.selections[key]
            ])
        return self.cluster.selections[key].to_record()

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, str):
            raise TypeError("Only column assignment by field name is supported.")

        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            if len(self.cluster) == 0:
                raise IndexError("Cannot assign a scalar column on an empty FitSelectionCluster.")
            values = [value] * len(self.cluster)
        else:
            values = list(value)
            if len(values) == 0 and len(self.cluster) == 0:
                return
            if len(values) != len(self.cluster):
                raise ValueError("Column length must match the number of selections")

        for selection, item in zip(self.cluster, values):
            selection.fields[key] = item

    def __iter__(self) -> Iterator[Dataset]:
        for selection in self.cluster:
            yield selection.to_record()

    def get_column_names(self) -> list[str]:
        return self.cluster.to_datacluster().get_column_names()

    def to_datacluster(self) -> DataCluster:
        return self.cluster.to_datacluster()


@dataclass(frozen=True)
class FitSelectionCluster(Sequence[FitSelection]):
    selections: list[FitSelection] = field(default_factory=list)
    fieldtable: FitSelectionFieldTable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fieldtable", FitSelectionFieldTable(self))

    def __getitem__(self, index):
        return self.selections[index]

    def __len__(self) -> int:
        return len(self.selections)

    def __iter__(self) -> Iterator[FitSelection]:
        return iter(self.selections)

    @property
    def analyses(self) -> list[FitAnalysis]:
        return [selection.analysis for selection in self.selections]

    @property
    def fit_results(self) -> list["FitResult"]:
        return [selection.fit_result for selection in self.selections]

    @property
    def fields(self) -> list[Dataset]:
        return [selection.fields for selection in self.selections]

    @property
    def extras(self) -> list[Dataset]:
        return self.fields

    @property
    def data(self) -> FitSelectionFieldTable:
        return self.fieldtable

    def to_datacluster(
        self,
        *,
        fields: Mapping[str, Any] | Dataset | None = None,
        on_conflict: RecordConflictMode = "raise",
    ) -> DataCluster:
        return DataCluster([
            selection.to_record(
                fields=fields,
                on_conflict=on_conflict,
            )
            for selection in self.selections
        ])

    def by_ref(self, ref: SelectionRef) -> FitSelection:
        matches = [selection for selection in self.selections if selection.ref == ref]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous selection reference '{ref}'.")
        raise KeyError(f"Unknown selection reference '{ref}'.")

    def merge(self, *others: MergeableSelection) -> "FitSelectionCluster":
        selections = list(self.selections)
        for other in others:
            if isinstance(other, FitSelection):
                selections.append(other)
                continue
            selections.extend(other.selections)
        return FitSelectionCluster(selections)

    @classmethod
    def concat(
        cls,
        collections: Iterable["FitSelectionCluster" | Iterable[FitSelection]],
    ) -> "FitSelectionCluster":
        selections: list[FitSelection] = []
        for collection in collections:
            if isinstance(collection, FitSelectionCluster):
                selections.extend(collection.selections)
            else:
                selections.extend(list(collection))
        return cls(selections)
