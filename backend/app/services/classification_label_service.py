import re
import unicodedata
from dataclasses import dataclass

from core.task_compatibility import (
    CLASSIFICATION_TASK_TYPE,
    require_model_task_type,
)
from schemas.classification_metrics import (
    ClassReference,
    LabelResolutionSummary,
    UnresolvedLabel,
)
from schemas.datasets import (
    DatasetLabelResolutionResponse,
    DatasetResponse,
)
from schemas.models import (
    ModelClassCatalog,
    ModelClassInfo,
)
from services.dataset_service import get_dataset
from services.model_service import get_active_model_instance


_NON_WORD_PATTERN = re.compile(
    r"[^\w\s]",
    flags=re.UNICODE,
)


def normalize_classification_label(label: str) -> str:
    """
    Normaliza una etiqueta para compararla con el catálogo.

    La normalización:
    - aplica Unicode NFKC;
    - ignora mayúsculas y minúsculas;
    - considera guiones y guiones bajos como espacios;
    - elimina puntuación;
    - colapsa espacios repetidos.

    No realiza traducción ni coincidencia difusa para evitar
    asignaciones incorrectas.
    """
    normalized = unicodedata.normalize(
        "NFKC",
        label,
    ).casefold()

    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")

    normalized = _NON_WORD_PATTERN.sub(
        " ",
        normalized,
    )

    return " ".join(normalized.split())


def _extract_class_aliases(
    class_label: str,
) -> tuple[str, ...]:
    """
    Extrae los alias separados por comas del catálogo ImageNet.

    Ejemplo:
        "tabby, tabby cat"
        → ("tabby", "tabby cat")
    """
    aliases = []

    for raw_alias in class_label.split(","):
        alias = raw_alias.strip()

        if alias and alias not in aliases:
            aliases.append(alias)

    return tuple(aliases)


@dataclass(frozen=True, slots=True)
class ClassificationLabelMatch:
    """
    Resultado de intentar resolver una etiqueta individual.
    """

    provided_label: str
    normalized_label: str

    class_reference: ClassReference | None = None
    matched_alias: str | None = None

    reason: str | None = None
    candidate_classes: tuple[ClassReference, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.class_reference is not None


@dataclass(frozen=True, slots=True)
class ResolvedDatasetLabel:
    """
    Etiqueta de una imagen resuelta a un índice de clase.
    """

    image_id: str
    filename: str
    provided_label: str

    class_reference: ClassReference
    matched_alias: str


@dataclass(frozen=True, slots=True)
class DatasetLabelResolution:
    """
    Resultado interno completo de resolver un dataset.

    La evaluación avanzada utilizará resolved_by_image_id para
    obtener el índice real de cada imagen.
    """

    dataset_id: str
    model_id: str
    model_name: str

    resolved: tuple[ResolvedDatasetLabel, ...]
    summary: LabelResolutionSummary

    @property
    def resolved_by_image_id(
        self,
    ) -> dict[str, ResolvedDatasetLabel]:
        return {label.image_id: label for label in self.resolved}


class ClassificationLabelResolver:
    """
    Resuelve etiquetas libres contra un catálogo canónico.

    No utiliza fuzzy matching porque una coincidencia aproximada
    podría asociar una imagen a una clase incorrecta.
    """

    def __init__(
        self,
        class_labels: tuple[str, ...],
    ):
        if not class_labels:
            raise ValueError("Classification model has no class labels")

        if any(not class_label.strip() for class_label in class_labels):
            raise ValueError("Classification class labels cannot be empty")

        self._class_labels = tuple(class_label.strip() for class_label in class_labels)

        self._aliases_by_index: dict[
            int,
            tuple[str, ...],
        ] = {}

        self._indices_by_normalized_label: dict[
            str,
            set[int],
        ] = {}

        self._build_index()

    def _build_index(self) -> None:
        for index, class_label in enumerate(self._class_labels):
            aliases = _extract_class_aliases(class_label)

            self._aliases_by_index[index] = aliases

            searchable_labels = (
                class_label,
                *aliases,
            )

            for searchable_label in searchable_labels:
                normalized_label = normalize_classification_label(searchable_label)

                if not normalized_label:
                    continue

                self._indices_by_normalized_label.setdefault(
                    normalized_label,
                    set(),
                ).add(index)

    def get_catalog(
        self,
    ) -> tuple[ModelClassInfo, ...]:
        class_catalog = []

        for index, class_label in enumerate(self._class_labels):
            aliases = [
                alias
                for alias in self._aliases_by_index[index]
                if normalize_classification_label(alias)
                != normalize_classification_label(class_label)
            ]

            class_catalog.append(
                ModelClassInfo(
                    index=index,
                    label=class_label,
                    aliases=aliases,
                )
            )

        return tuple(class_catalog)

    def resolve(
        self,
        provided_label: str,
    ) -> ClassificationLabelMatch:
        normalized_label = normalize_classification_label(provided_label)

        if not normalized_label:
            return ClassificationLabelMatch(
                provided_label=provided_label,
                normalized_label=normalized_label,
                reason="Label is empty",
            )

        candidate_indices = sorted(
            self._indices_by_normalized_label.get(
                normalized_label,
                set(),
            )
        )

        candidate_classes = tuple(
            ClassReference(
                index=index,
                label=self._class_labels[index],
            )
            for index in candidate_indices
        )

        if not candidate_indices:
            return ClassificationLabelMatch(
                provided_label=provided_label,
                normalized_label=normalized_label,
                reason=("Label does not match any model class"),
            )

        if len(candidate_indices) > 1:
            return ClassificationLabelMatch(
                provided_label=provided_label,
                normalized_label=normalized_label,
                reason=("Label matches multiple model classes"),
                candidate_classes=candidate_classes,
            )

        class_index = candidate_indices[0]
        class_label = self._class_labels[class_index]

        searchable_labels = (
            class_label,
            *self._aliases_by_index[class_index],
        )

        matched_alias = next(
            (
                alias
                for alias in searchable_labels
                if normalize_classification_label(alias) == normalized_label
            ),
            class_label,
        )

        return ClassificationLabelMatch(
            provided_label=provided_label,
            normalized_label=normalized_label,
            class_reference=ClassReference(
                index=class_index,
                label=class_label,
            ),
            matched_alias=matched_alias,
            candidate_classes=candidate_classes,
        )


def build_label_resolver_for_model(
    model,
    endpoint: str,
) -> ClassificationLabelResolver:
    require_model_task_type(
        model=model,
        expected_task_type=CLASSIFICATION_TASK_TYPE,
        endpoint=endpoint,
    )

    return ClassificationLabelResolver(model.get_class_labels())


def resolve_dataset_labels(
    dataset: DatasetResponse,
    model,
    endpoint: str = "/evaluate",
) -> DatasetLabelResolution:
    """
    Resuelve las etiquetas no vacías del dataset.

    Las imágenes sin etiqueta no cuentan como etiquetas no resueltas.
    """
    resolver = build_label_resolver_for_model(
        model=model,
        endpoint=endpoint,
    )

    resolved_labels = []
    unresolved_labels = []

    labeled_images = 0

    for image in dataset.images:
        if image.label is None:
            continue

        provided_label = image.label.strip()

        if not provided_label:
            continue

        labeled_images += 1

        match = resolver.resolve(provided_label)

        if match.resolved:
            resolved_labels.append(
                ResolvedDatasetLabel(
                    image_id=image.id,
                    filename=image.filename,
                    provided_label=provided_label,
                    class_reference=(match.class_reference),
                    matched_alias=(match.matched_alias or provided_label),
                )
            )

            continue

        unresolved_labels.append(
            UnresolvedLabel(
                image_id=image.id,
                filename=image.filename,
                provided_label=provided_label,
                reason=(match.reason or "Label could not be resolved"),
                candidate_classes=list(match.candidate_classes),
            )
        )

    resolved_count = len(resolved_labels)
    unresolved_count = len(unresolved_labels)

    coverage = resolved_count / labeled_images if labeled_images else 0.0

    summary = LabelResolutionSummary(
        labeled_images=labeled_images,
        resolved_labels=resolved_count,
        unresolved_labels=unresolved_count,
        coverage=round(coverage, 4),
        unresolved=unresolved_labels,
    )

    return DatasetLabelResolution(
        dataset_id=dataset.id,
        model_id=model.id,
        model_name=model.name,
        resolved=tuple(resolved_labels),
        summary=summary,
    )


def get_active_model_class_catalog() -> ModelClassCatalog:
    model = get_active_model_instance()

    resolver = build_label_resolver_for_model(
        model=model,
        endpoint="/models/active/classes",
    )

    classes = list(resolver.get_catalog())

    return ModelClassCatalog(
        model_id=model.id,
        model_name=model.name,
        total_classes=len(classes),
        classes=classes,
    )


async def get_active_dataset_label_resolution(
    dataset_id: str,
) -> DatasetLabelResolutionResponse:
    dataset = await get_dataset(dataset_id)
    model = get_active_model_instance()

    resolution = resolve_dataset_labels(
        dataset=dataset,
        model=model,
        endpoint=("/datasets/{dataset_id}/label-resolution"),
    )

    return DatasetLabelResolutionResponse(
        dataset_id=resolution.dataset_id,
        model_id=resolution.model_id,
        model_name=resolution.model_name,
        summary=resolution.summary,
    )
