import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Protocol


class DetectionClassCatalogError(ValueError):
    """Indica que un catálogo de clases de detección es inválido."""


class DetectionCategoryResolutionError(ValueError):
    """Indica que una categoría de entrada es estructuralmente inválida."""


def normalize_detection_category_name(name: str) -> str:
    """
    Normaliza nombres sin inventar equivalencias semánticas.

    Aplica Unicode NFKC, ``casefold``, trata guiones y guiones bajos como
    espacios y colapsa espacios repetidos. No elimina otra puntuación ni
    realiza fuzzy matching, traducción o stemming.
    """
    if not isinstance(name, str) or not name.strip():
        raise DetectionCategoryResolutionError(
            "detection category name must be a non-empty string"
        )

    normalized = unicodedata.normalize("NFKC", name).casefold()
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())

    if not normalized:
        raise DetectionCategoryResolutionError(
            "normalized detection category name must not be empty"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class DetectionClass:
    """Clase canónica de un detector."""

    class_id: int
    name: str

    def __post_init__(self) -> None:
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, Integral):
            raise DetectionClassCatalogError("class_id must be an integer")

        class_id = int(self.class_id)
        if class_id < 0:
            raise DetectionClassCatalogError("class_id must not be negative")
        if not isinstance(self.name, str) or not self.name.strip():
            raise DetectionClassCatalogError("class name must be a non-empty string")

        name = self.name.strip()
        try:
            normalize_detection_category_name(name)
        except DetectionCategoryResolutionError as error:
            raise DetectionClassCatalogError(str(error)) from error

        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class DetectionClassNameCollision:
    """Clases distintas que comparten el mismo nombre normalizado."""

    normalized_name: str
    classes: tuple[DetectionClass, ...]


@dataclass(frozen=True, slots=True)
class DetectionClassCatalog:
    """Catálogo completo, inmutable y agnóstico de un detector."""

    model_id: str
    model_name: str
    classes: tuple[DetectionClass, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise DetectionClassCatalogError("model_id must be a non-empty string")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise DetectionClassCatalogError("model_name must be a non-empty string")

        try:
            classes = tuple(self.classes)
        except TypeError as error:
            raise DetectionClassCatalogError("classes must be an iterable") from error

        if not classes:
            raise DetectionClassCatalogError(
                "detection class catalog must not be empty"
            )
        if any(not isinstance(model_class, DetectionClass) for model_class in classes):
            raise DetectionClassCatalogError(
                "classes must contain only DetectionClass values"
            )

        class_ids = [model_class.class_id for model_class in classes]
        if len(class_ids) != len(set(class_ids)):
            raise DetectionClassCatalogError("detection class IDs must be unique")

        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(
            self,
            "classes",
            tuple(sorted(classes, key=lambda model_class: model_class.class_id)),
        )

    @property
    def class_by_id(self) -> dict[int, DetectionClass]:
        return {model_class.class_id: model_class for model_class in self.classes}

    @property
    def normalized_name_collisions(
        self,
    ) -> tuple[DetectionClassNameCollision, ...]:
        classes_by_name: dict[str, list[DetectionClass]] = defaultdict(list)
        for model_class in self.classes:
            normalized_name = normalize_detection_category_name(model_class.name)
            classes_by_name[normalized_name].append(model_class)

        return tuple(
            DetectionClassNameCollision(
                normalized_name=normalized_name,
                classes=tuple(
                    sorted(
                        candidates,
                        key=lambda candidate: candidate.class_id,
                    )
                ),
            )
            for normalized_name, candidates in sorted(classes_by_name.items())
            if len(candidates) > 1
        )


class CanonicalDetectionCategory(Protocol):
    internal_id: str
    external_id: int
    name: str


class DetectionCategoryResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


class DetectionCategoryResolutionReason(str, Enum):
    NO_MATCHING_MODEL_CLASS = "no_matching_model_class"
    AMBIGUOUS_MODEL_CLASS_NAME = "ambiguous_model_class_name"
    MODEL_CLASS_CLAIMED_BY_MULTIPLE_CATEGORIES = (
        "model_class_claimed_by_multiple_categories"
    )


@dataclass(frozen=True, slots=True)
class DetectionCategoryMatch:
    """Resultado trazable de resolver una categoría anotada."""

    category_internal_id: str
    coco_category_id: int
    category_name: str
    normalized_category_name: str
    status: DetectionCategoryResolutionStatus
    resolved_class: DetectionClass | None = None
    reason: DetectionCategoryResolutionReason | None = None
    candidate_classes: tuple[DetectionClass, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.category_internal_id, str)
            or not self.category_internal_id.strip()
        ):
            raise DetectionCategoryResolutionError(
                "category_internal_id must be a non-empty string"
            )
        if isinstance(self.coco_category_id, bool) or not isinstance(
            self.coco_category_id,
            Integral,
        ):
            raise DetectionCategoryResolutionError(
                "coco_category_id must be an integer"
            )
        if not isinstance(self.category_name, str) or not self.category_name.strip():
            raise DetectionCategoryResolutionError(
                "category_name must be a non-empty string"
            )
        expected_normalized_name = normalize_detection_category_name(self.category_name)
        if self.normalized_category_name != expected_normalized_name:
            raise DetectionCategoryResolutionError(
                "normalized_category_name does not match category_name"
            )
        if not isinstance(self.status, DetectionCategoryResolutionStatus):
            raise DetectionCategoryResolutionError(
                "status must be a DetectionCategoryResolutionStatus"
            )

        candidates = tuple(self.candidate_classes)
        if any(not isinstance(candidate, DetectionClass) for candidate in candidates):
            raise DetectionCategoryResolutionError(
                "candidate_classes must contain only DetectionClass values"
            )
        candidate_ids = [candidate.class_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise DetectionCategoryResolutionError(
                "candidate_classes must contain unique class IDs"
            )

        if self.status is DetectionCategoryResolutionStatus.RESOLVED:
            if not isinstance(self.resolved_class, DetectionClass):
                raise DetectionCategoryResolutionError(
                    "resolved match must reference one DetectionClass"
                )
            if self.reason is not None or candidates:
                raise DetectionCategoryResolutionError(
                    "resolved match must not contain a reason or candidates"
                )
        else:
            if self.resolved_class is not None or self.reason is None:
                raise DetectionCategoryResolutionError(
                    "unresolved match must contain a reason and no resolved class"
                )
            if self.status is DetectionCategoryResolutionStatus.UNRESOLVED:
                if (
                    self.reason
                    is not DetectionCategoryResolutionReason.NO_MATCHING_MODEL_CLASS
                ):
                    raise DetectionCategoryResolutionError(
                        "unknown category has an invalid reason"
                    )
                if candidates:
                    raise DetectionCategoryResolutionError(
                        "unknown category must not contain candidates"
                    )
            elif self.status is DetectionCategoryResolutionStatus.AMBIGUOUS:
                if (
                    self.reason
                    is not DetectionCategoryResolutionReason.AMBIGUOUS_MODEL_CLASS_NAME
                ):
                    raise DetectionCategoryResolutionError(
                        "ambiguous category has an invalid reason"
                    )
                if len(candidates) < 2:
                    raise DetectionCategoryResolutionError(
                        "ambiguous category must contain multiple candidates"
                    )
            elif self.status is DetectionCategoryResolutionStatus.CONFLICT:
                if (
                    self.reason
                    is not DetectionCategoryResolutionReason.MODEL_CLASS_CLAIMED_BY_MULTIPLE_CATEGORIES
                ):
                    raise DetectionCategoryResolutionError(
                        "conflicting category has an invalid reason"
                    )
                if len(candidates) != 1:
                    raise DetectionCategoryResolutionError(
                        "conflicting category must contain its single candidate"
                    )

        object.__setattr__(
            self,
            "category_internal_id",
            self.category_internal_id.strip(),
        )
        object.__setattr__(self, "coco_category_id", int(self.coco_category_id))
        object.__setattr__(self, "category_name", self.category_name.strip())
        object.__setattr__(self, "candidate_classes", candidates)

    @property
    def resolved(self) -> bool:
        return self.status is DetectionCategoryResolutionStatus.RESOLVED


@dataclass(frozen=True, slots=True)
class DetectionCategoryResolution:
    """Resolución completa entre un snapshot y un catálogo de detector."""

    catalog: DetectionClassCatalog
    matches: tuple[DetectionCategoryMatch, ...]
    total_categories: int = field(init=False)
    resolved_count: int = field(init=False)
    unresolved_count: int = field(init=False)
    ambiguity_count: int = field(init=False)
    conflict_count: int = field(init=False)
    coverage: float = field(init=False)
    resolved_category_class_pairs: tuple[tuple[str, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, DetectionClassCatalog):
            raise DetectionCategoryResolutionError(
                "catalog must be a DetectionClassCatalog"
            )

        matches = tuple(self.matches)
        if any(not isinstance(match, DetectionCategoryMatch) for match in matches):
            raise DetectionCategoryResolutionError(
                "matches must contain only DetectionCategoryMatch values"
            )
        internal_ids = [match.category_internal_id for match in matches]
        if len(internal_ids) != len(set(internal_ids)):
            raise DetectionCategoryResolutionError(
                "matches must contain unique category_internal_id values"
            )

        catalog_classes = self.catalog.class_by_id
        resolved_class_ids: set[int] = set()
        validated_candidate_groups: set[tuple[int, str]] = set()

        for match in matches:
            resolved_class = match.resolved_class
            if resolved_class is not None:
                catalog_class = catalog_classes.get(resolved_class.class_id)
                if catalog_class != resolved_class:
                    raise DetectionCategoryResolutionError(
                        f"resolved class ID {resolved_class.class_id} for category "
                        f"'{match.category_internal_id}' is not present in catalog "
                        f"'{self.catalog.model_id}'"
                    )
                if (
                    normalize_detection_category_name(resolved_class.name)
                    != match.normalized_category_name
                ):
                    raise DetectionCategoryResolutionError(
                        f"resolved class ID {resolved_class.class_id} name does not "
                        f"match category '{match.category_internal_id}'"
                    )
                if resolved_class.class_id in resolved_class_ids:
                    raise DetectionCategoryResolutionError(
                        f"resolved class ID {resolved_class.class_id} is claimed by "
                        "multiple dataset categories"
                    )
                resolved_class_ids.add(resolved_class.class_id)

            if not match.candidate_classes:
                continue

            candidate_group_key = (
                id(match.candidate_classes),
                match.normalized_category_name,
            )
            if candidate_group_key in validated_candidate_groups:
                continue

            candidate_ids: set[int] = set()
            for candidate in match.candidate_classes:
                if candidate.class_id in candidate_ids:
                    raise DetectionCategoryResolutionError(
                        f"candidate class ID {candidate.class_id} is duplicated for "
                        f"category '{match.category_internal_id}'"
                    )
                candidate_ids.add(candidate.class_id)

                catalog_class = catalog_classes.get(candidate.class_id)
                if catalog_class != candidate:
                    raise DetectionCategoryResolutionError(
                        f"candidate class ID {candidate.class_id} for category "
                        f"'{match.category_internal_id}' is not present in catalog "
                        f"'{self.catalog.model_id}'"
                    )
                if (
                    normalize_detection_category_name(candidate.name)
                    != match.normalized_category_name
                ):
                    raise DetectionCategoryResolutionError(
                        f"candidate class ID {candidate.class_id} name does not "
                        f"match category '{match.category_internal_id}'"
                    )

            validated_candidate_groups.add(candidate_group_key)

        total_categories = len(matches)
        resolved_count = sum(match.resolved for match in matches)
        ambiguity_count = sum(
            match.status is DetectionCategoryResolutionStatus.AMBIGUOUS
            for match in matches
        )
        conflict_count = sum(
            match.status is DetectionCategoryResolutionStatus.CONFLICT
            for match in matches
        )
        resolved_pairs = tuple(
            (match.category_internal_id, match.resolved_class.class_id)
            for match in matches
            if match.resolved_class is not None
        )

        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "total_categories", total_categories)
        object.__setattr__(self, "resolved_count", resolved_count)
        object.__setattr__(
            self,
            "unresolved_count",
            total_categories - resolved_count,
        )
        object.__setattr__(self, "ambiguity_count", ambiguity_count)
        object.__setattr__(self, "conflict_count", conflict_count)
        object.__setattr__(
            self,
            "coverage",
            resolved_count / total_categories if total_categories else 0.0,
        )
        object.__setattr__(
            self,
            "resolved_category_class_pairs",
            resolved_pairs,
        )

    @property
    def class_id_by_category_internal_id(self) -> dict[str, int]:
        return dict(self.resolved_category_class_pairs)


@dataclass(frozen=True, slots=True)
class _PreparedCategory:
    internal_id: str
    external_id: int
    name: str
    normalized_name: str


def _prepare_categories(
    categories: Sequence[CanonicalDetectionCategory],
) -> tuple[_PreparedCategory, ...]:
    prepared = []
    seen_internal_ids = set()

    for category in categories:
        internal_id = category.internal_id
        external_id = category.external_id
        name = category.name

        if not isinstance(internal_id, str) or not internal_id.strip():
            raise DetectionCategoryResolutionError(
                "category internal_id must be a non-empty string"
            )
        normalized_internal_id = internal_id.strip()
        if normalized_internal_id in seen_internal_ids:
            raise DetectionCategoryResolutionError(
                f"duplicate category internal_id '{normalized_internal_id}'"
            )
        if isinstance(external_id, bool) or not isinstance(external_id, Integral):
            raise DetectionCategoryResolutionError(
                "category external_id must be an integer"
            )
        if not isinstance(name, str) or not name.strip():
            raise DetectionCategoryResolutionError(
                "category name must be a non-empty string"
            )

        prepared.append(
            _PreparedCategory(
                internal_id=normalized_internal_id,
                external_id=int(external_id),
                name=name.strip(),
                normalized_name=normalize_detection_category_name(name),
            )
        )
        seen_internal_ids.add(normalized_internal_id)

    return tuple(
        sorted(
            prepared,
            key=lambda category: (
                category.normalized_name,
                category.external_id,
                category.internal_id,
            ),
        )
    )


def _build_classes_by_normalized_name(
    catalog: DetectionClassCatalog,
) -> Mapping[str, tuple[DetectionClass, ...]]:
    classes_by_name: dict[str, list[DetectionClass]] = defaultdict(list)
    for model_class in catalog.classes:
        normalized_name = normalize_detection_category_name(model_class.name)
        classes_by_name[normalized_name].append(model_class)

    return MappingProxyType(
        {
            normalized_name: tuple(candidates)
            for normalized_name, candidates in classes_by_name.items()
        }
    )


def resolve_detection_categories(
    *,
    catalog: DetectionClassCatalog,
    categories: Sequence[CanonicalDetectionCategory],
) -> DetectionCategoryResolution:
    """Resuelve por nombre con un índice de construcción y consulta ``O(C + M)``."""
    if not isinstance(catalog, DetectionClassCatalog):
        raise DetectionCategoryResolutionError(
            "catalog must be a DetectionClassCatalog"
        )

    prepared_categories = _prepare_categories(categories)
    classes_by_normalized_name = _build_classes_by_normalized_name(catalog)

    unresolved_matches = []
    provisional_matches: list[tuple[_PreparedCategory, DetectionClass]] = []
    categories_by_class_id: dict[int, list[_PreparedCategory]] = defaultdict(list)

    for category in prepared_categories:
        candidates = classes_by_normalized_name.get(category.normalized_name, ())
        if not candidates:
            unresolved_matches.append(
                DetectionCategoryMatch(
                    category_internal_id=category.internal_id,
                    coco_category_id=category.external_id,
                    category_name=category.name,
                    normalized_category_name=category.normalized_name,
                    status=DetectionCategoryResolutionStatus.UNRESOLVED,
                    reason=(DetectionCategoryResolutionReason.NO_MATCHING_MODEL_CLASS),
                )
            )
            continue
        if len(candidates) > 1:
            unresolved_matches.append(
                DetectionCategoryMatch(
                    category_internal_id=category.internal_id,
                    coco_category_id=category.external_id,
                    category_name=category.name,
                    normalized_category_name=category.normalized_name,
                    status=DetectionCategoryResolutionStatus.AMBIGUOUS,
                    reason=(
                        DetectionCategoryResolutionReason.AMBIGUOUS_MODEL_CLASS_NAME
                    ),
                    candidate_classes=candidates,
                )
            )
            continue

        model_class = candidates[0]
        provisional_matches.append((category, model_class))
        categories_by_class_id[model_class.class_id].append(category)

    resolved_or_conflicting_matches = []
    for category, model_class in provisional_matches:
        claims = categories_by_class_id[model_class.class_id]
        if len(claims) > 1:
            resolved_or_conflicting_matches.append(
                DetectionCategoryMatch(
                    category_internal_id=category.internal_id,
                    coco_category_id=category.external_id,
                    category_name=category.name,
                    normalized_category_name=category.normalized_name,
                    status=DetectionCategoryResolutionStatus.CONFLICT,
                    reason=(
                        DetectionCategoryResolutionReason.MODEL_CLASS_CLAIMED_BY_MULTIPLE_CATEGORIES
                    ),
                    candidate_classes=(model_class,),
                )
            )
        else:
            resolved_or_conflicting_matches.append(
                DetectionCategoryMatch(
                    category_internal_id=category.internal_id,
                    coco_category_id=category.external_id,
                    category_name=category.name,
                    normalized_category_name=category.normalized_name,
                    status=DetectionCategoryResolutionStatus.RESOLVED,
                    resolved_class=model_class,
                )
            )

    matches = tuple(
        sorted(
            (*unresolved_matches, *resolved_or_conflicting_matches),
            key=lambda match: (
                match.normalized_category_name,
                match.coco_category_id,
                match.category_internal_id,
            ),
        )
    )
    return DetectionCategoryResolution(
        catalog=catalog,
        matches=matches,
    )
