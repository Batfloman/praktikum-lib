from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from ...structs.dataCluster import DataCluster
from ...structs.dataset import Dataset
from .analysis import ComponentFitAnalysis, FitAnalysis, RecordConflictMode, _merge_record_data

if TYPE_CHECKING:
    from ..fitResult import FIT_METHODS
    from .session import FitSession
    from ..fitResult import FitResult

type SelectionRef = str | int


class SelectionOptions(TypedDict):
    ref: SelectionRef
    component: NotRequired[SelectionRef | None]
    extra: NotRequired[Mapping[str, Any] | Dataset | None]
    rename: NotRequired[Mapping[str, str] | None]
    auto_fit: NotRequired[bool]
    method: NotRequired[FIT_METHODS | None]


type SelectionSpec = SelectionRef | SelectionOptions
type SelectionIterable = Iterable[SelectionSpec]
type MergeableSelection = FitSelection | FitSelectionCluster

def _normalize_extra(extra: Mapping[str, Any] | Dataset | None) -> Dataset | None:
    if extra is None:
        return None
    if isinstance(extra, Dataset):
        return extra.copy()
    return Dataset(extra)


def _apply_rename(
    record: Dataset,
    rename: Mapping[str, str] | None,
) -> Dataset:
    if rename is None:
        return record.copy()
    return record.rename(**dict(rename))


@dataclass(frozen=True)
class FitSelection:
    session: "FitSession"
    ref: SelectionRef
    analysis: FitAnalysis
    component_ref: SelectionRef | None = None
    extra: Dataset | None = None
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

    def to_record(
        self,
        *,
        extra: Mapping[str, Any] | Dataset | None = None,
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
        resolved_extra = _merge_record_data(
            self.extra or Dataset(),
            _normalize_extra(extra),
            on_conflict=on_conflict,
        )
        return _merge_record_data(
            resolved_record,
            resolved_extra,
            on_conflict=on_conflict,
        )

    def merge(self, *others: MergeableSelection) -> "FitSelectionCluster":
        return FitSelectionCluster([self]).merge(*others)


@dataclass(frozen=True)
class FitSelectionCluster(Sequence[FitSelection]):
    selections: list[FitSelection] = field(default_factory=list)

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
    def extras(self) -> list[Dataset | None]:
        return [selection.extra for selection in self.selections]

    def to_datacluster(
        self,
        *,
        extra: Mapping[str, Any] | Dataset | None = None,
        on_conflict: RecordConflictMode = "raise",
    ) -> DataCluster:
        return DataCluster([
            selection.to_record(extra=extra, on_conflict=on_conflict)
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
