# Informe: métricas COCO supervisadas para evaluación de detección

## 1. Resultado

- Estado: completado.
- `HEAD` inicial y final, sin commit:
  `995fc040af2a675e32208fd7b20790ed18a65486`.
- Commit inspeccionado:
  `995fc04 feat(backend): se ha añadido resolución de categorías mediante detección canónica`.
- El árbol estaba limpio al iniciar la sub-issue.
- Se conservaron las piezas previas de catálogo, predicción canónica,
  snapshots, CAS y resolución de categorías.
- Se añadió únicamente `pycocotools==2.0.11`.
- No se modificaron Dockerfile, Compose, migraciones, frontend,
  autenticación, autorización ni ownership.
- No se persistieron métricas ni se crearon artefactos visuales.
- No se hizo commit ni push.

### Corrección posterior de aislamiento legacy

La revisión posterior encontró que la primera implementación no aislaba por
completo el contrato legacy:

- `nms_iou_threshold` y `max_detections_per_image` eran entrada del cliente;
- esos valores alimentaban la inferencia de la que también se derivaban
  `results`, `summary`, `attack_success` y `attack_success_rate`;
- el adaptador legacy usaba `confidence >= 0.25`, mientras Ultralytics 8.4.104
  aplica históricamente un corte estricto `confidence > 0.25`;
- incluso sin métricas supervisadas se invocaba
  `detect_for_evaluation()`;
- un snapshot activo solicitaba el catálogo antes de comprobar si había
  imágenes revisadas.

La causa raíz fue tratar toda la configuración científica como configurable
sin separar cuáles de sus parámetros forman parte de la semántica histórica
de la inferencia. La corrección fija NMS y `max_det`, introduce una frontera
legacy con nombre explícito y bifurca el camino de inferencia después de
preparar la disponibilidad supervisada.

La evaluación de detección devuelve ahora, además del contrato legacy, una
sección `supervised` con:

- disponibilidad y motivo tipado;
- cobertura de anotaciones;
- configuración efectiva;
- análisis COCO original y adversario;
- deltas `original - adversarial`;
- AP/AR agregados, por tamaño y por clase;
- curvas PR agregadas y por clase;
- TP, FP, FN, precision, recall y F1 en un punto configurable.

## 2. Comprobaciones iniciales

Los comandos ejecutados antes de modificar código fueron:

```text
git rev-parse HEAD
git status --short
git log -1 --oneline
```

Resultados:

```text
995fc040af2a675e32208fd7b20790ed18a65486
<sin salida; árbol limpio>
995fc04 feat(backend): se ha añadido resolución de categorías mediante detección canónica
```

Solo se encontró un `AGENTS.md` bajo
`frontend/node_modules/recharts`; no es aplicable al backend.

Se verificó que `HEAD` ya contenía:

- `Detection`, `DetectionBox` y `DetectionResult`;
- `BaseObjectDetectionModel.get_detection_class_catalog()`;
- `YOLODetectionModel.get_detection_class_catalog()`;
- `build_ultralytics_class_catalog()`;
- `DetectionClassCatalog`;
- `resolve_detection_categories()`;
- `category_internal_id → model class_id`;
- estados `resolved`, `unresolved`, `ambiguous` y `conflict`.

No se reimplementó ninguna de estas piezas.

El baseline focalizado previo a los cambios ejecutó 159 pruebas:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_evaluation_service.py \
  tests/test_detection_evaluation.py \
  tests/test_detection_metrics.py \
  tests/test_detection_category_resolution.py \
  tests/test_detection_class_catalog.py \
  tests/test_detection_prediction_contract.py \
  tests/test_yolo_detection_adapter.py \
  tests/test_models.py \
  tests/test_coco_annotation_importer.py \
  tests/test_dataset_coco_annotations.py
```

Resultado: código 0, `159 passed in 14.19s`, sin fallos, skips ni warnings
informados.

### Inspección inicial de la ronda correctiva

Antes de esta corrección se ejecutaron:

```text
git rev-parse HEAD
git status --short
git diff --stat
git diff
git ls-files --others --exclude-standard
```

`HEAD` seguía en
`995fc040af2a675e32208fd7b20790ed18a65486`. Se inspeccionó el diff completo;
contenía exclusivamente la implementación sin commit descrita en este
informe. `git status --short` mostraba diez archivos tracked modificados y
ocho archivos nuevos, incluido este informe. Todos se preservaron.

Baseline focalizado de la ronda:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_coco_detection_metrics.py \
  tests/test_detection_supervised_metrics_contract.py \
  tests/test_detection_annotation_snapshot_reader.py \
  tests/test_detection_evaluation_service.py \
  tests/test_detection_evaluation.py \
  tests/test_yolo_detection_adapter.py \
  tests/test_unexpected_service_errors.py
```

Resultado: código 0, 74 recopiladas, `74 passed in 1.15s`, sin fallos, skips
ni warnings informados.

## 3. Investigación técnica

Fecha de consulta: **2026-07-23**.

### 3.1 Versiones reales

Antes de añadir la dependencia:

```text
docker compose exec backend python -m pip show pycocotools
```

Resultado: código 1; el paquete no estaba instalado.

Tras reconstruir el backend:

```text
Name: pycocotools
Version: 2.0.11
Location: /usr/local/lib/python3.10/site-packages
Requires: numpy
```

```text
Name: ultralytics
Version: 8.4.104
Location: /usr/local/lib/python3.10/site-packages
```

`pycocotools` 2.0.11 declara Python `>=3.9` y dispone de distribución para el
Python 3.10 usado por el contenedor. No se añadió otra biblioteca métrica.

### 3.2 COCOeval instalado

La inspección del código instalado y una ejecución directa confirmaron:

```text
iouThrs       [0.50, 0.55, ..., 0.95]
recThrs_count 101
areaRngLbl    ['all', 'small', 'medium', 'large']
maxDets       [1, 10, 100]
precision     [T, R, K, A, M]
recall        [T, K, A, M]
```

Conclusiones:

- `evaluate()` produce `evalImgs` y el matching oficial por imagen,
  categoría y área;
- `accumulate()` produce los tensores interpolados de precision y recall;
- `summarize()` usa esos tensores y representa ausencia mediante `-1`;
- las detecciones se ordenan por score de forma estable;
- `gtIgnore`, `dtIgnore`, `gtMatches` y `dtMatches` son la fuente correcta
  para el punto de operación;
- `iscrowd` participa en la política oficial de matching/ignore;
- no debe incluirse `-1` como cero en ninguna media.

Se comprobó además el caso vacío de la versión instalada:

```text
ground_truth.loadRes([])
Loading and preparing results...
IndexError list index out of range
```

Por ello el adaptador construye directamente un objeto COCO de detecciones
vacío y no depende de `COCO.loadRes([])`.

### 3.3 Ultralytics instalado

`DEFAULT_CFG` de 8.4.104 informó:

```text
conf=None
iou=0.7
max_det=300
```

La implementación instalada de `BasePredictor.__init__` convierte
`conf=None` en `0.25`. Por tanto, la inferencia interactiva actual descarta el
ranking inferior a 0.25 y no es adecuada para presentar mAP/curvas PR como
COCO estándar.

Decisión:

- inferencia científica por defecto: `conf=0.001`, `iou=0.7`,
  `max_det=300`; solo `conf` puede configurarse hasta `0.25`;
- punto de operación por defecto: confidence `0.25`, IoU `0.50`;
- COCOeval conserva `maxDets=[1, 10, 100]`;
- el contrato legacy se obtiene con
  `build_legacy_detection_result()` y el corte estricto
  `confidence > 0.25`, sin repetir inferencia;
- `score_floor` no puede superar `0.25`, para que una configuración del
  cliente no elimine predicciones necesarias para reproducir el resultado
  legacy;
- `confidence_threshold` no puede ser inferior al `score_floor`;
- `nms_iou_threshold=0.7` y `max_detections_per_image=300` son constantes
  internas, no entrada configurable;
- el request rechaza esos nombres como campos extra;
- original y adversaria reciben exactamente la misma configuración.

La inferencia normal de `detect()`, y por tanto `/detect`,
`/detect/attack` y otros consumidores interactivos, sigue invocando
Ultralytics sin overrides.

### 3.4 Fuentes primarias

- [COCOeval oficial](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py):
  evaluación, acumulación, dimensiones, ignores, áreas y `maxDets`.
- [COCO API oficial](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py):
  construcción de índices y comportamiento de `loadRes`.
- [Formato de datos COCO](https://cocodataset.org/#format-data):
  imágenes, categorías, anotaciones, `bbox`, `area` e `iscrowd`.
- [pycocotools 2.0.11 en PyPI](https://pypi.org/project/pycocotools/):
  versión y compatibilidad de Python.
- [Configuración oficial de Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml):
  opciones `conf`, `iou` y `max_det`.
- [Código oficial de predicción de Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/predictor.py):
  aplicación de `conf=0.25` cuando no se proporciona un valor.
- [Documentación oficial de configuración](https://docs.ultralytics.com/usage/cfg/):
  semántica de argumentos de predicción.

## 4. Arquitectura implementada

```text
dataset_revision + active_annotation_snapshot_id
                     │
                     ▼
       lectura optimista y reintento acotado
                     │
                     ▼
 MaterializedDetectionAnnotationSnapshot (inmutable)
                     │
       ┌─────────────┴──────────────┐
       ▼                            ▼
DetectionClassCatalog      imágenes revisadas + GT
       │                            │
       └── resolve una sola vez ────┘
                     │
                     ▼
          PreparedDetectionSupervisedEvaluation
                     │
        una inferencia original/adversaria
                     │
                     ▼
       contratos propios de ml/metrics
                     │
                     ▼
            COCOeval encapsulado
                     │
                     ▼
 schemas públicos + deltas + contrato legacy
```

### 4.1 Snapshot canónico autoconsistente

`read_detection_annotation_snapshot()`:

1. valida el ID de dataset;
2. lee `dataset_revision` y `active_annotation_snapshot_id`;
3. consulta categorías y objetos con los dos filtros obligatorios:
   `dataset_id + snapshot_id`;
4. vuelve a leer revisión y snapshot;
5. acepta únicamente si ambos siguen iguales;
6. reintenta como máximo tres veces;
7. devuelve un `ConflictError` tipado y accionable si el estado cambia
   repetidamente.

No lee el JSON COCO, no ejecuta el importador y no amplía `DatasetResponse`.
El snapshot materializado usa dataclasses congeladas, `slots`, tuplas y un
`frozenset` calculado de imágenes revisadas.

La materialización valida:

- documento de dataset, secuencia de imágenes y campos obligatorios;
- documentos de categorías y objetos con contexto seguro de tipo e ID;
- tipos de `_id`, `dataset_id`, `snapshot_id`, IDs COCO, nombres, cajas,
  `area` e `iscrowd`;
- imágenes y categorías sin IDs duplicados;
- referencias de objetos a imagen y categoría;
- estado de revisión;
- dimensiones autoritativas para imágenes revisadas;
- cajas dentro de geometría;
- área positiva e `iscrowd` válido;
- coherencia entre `reviewed_empty`, `reviewed_with_objects` y objetos.

Los fallos de PyMongo se encadenan como
`DetectionAnnotationSnapshotReadError`; los cambios concurrentes no se
confunden con indisponibilidad de métricas. Los campos ausentes producen
`DetectionAnnotationSnapshotError` y conservan el `KeyError` como
`__cause__`. Los tipos y relaciones incoherentes producen el mismo error
estructural específico, sin capturas amplias de errores de programación.

### 4.2 Universo y cobertura

- `reviewed_with_objects`: participa con sus objetos.
- `reviewed_empty`: participa como negativo real; sus predicciones pueden ser
  FP.
- `unreviewed`: no entra en COCOeval.
- El flujo legacy continúa evaluando todas las imágenes.
- Solo las categorías `resolved` participan.
- Las categorías `unresolved`, `ambiguous` y `conflict` y sus objetos se
  excluyen, reducen la cobertura y producen warnings deterministas.
- Todas las categorías resueltas aparecen por clase aunque tengan soporte
  cero; en ese caso AP, AR y curva son `null`.

La sección supervisada solo está disponible con snapshot activo, alguna
imagen revisada, alguna categoría resuelta y al menos un objeto resuelto no
ignorado. Los motivos tipados son:

- `no_annotation_snapshot`;
- `no_reviewed_images`;
- `no_resolved_categories`;
- `no_resolved_ground_truth_objects`.

### 4.3 Inferencia configurada

`DetectionEvaluationInferenceConfiguration` es un contrato interno agnóstico
del proveedor. Solo admite como entrada:

- `score_floor`, configurable en `[0, 0.25]`.

Expone además como configuración efectiva, pero no configurable:

- `nms_iou_threshold=0.7`;
- `max_detections_per_image=300`.

`BaseObjectDetectionModel.detect_for_evaluation()` declara una política
explícita: un detector que no implemente inferencia científica falla con
`NotImplementedError` en vez de usar silenciosamente defaults interactivos.

YOLO pasa los tres valores a Ultralytics y devuelve el mismo
`DetectionResult` canónico de precisión completa. `detect()` conserva su
llamada anterior sin argumentos.

`DetectionSupervisedConfiguration` solo permite cambiar `score_floor`,
`confidence_threshold` e `iou_threshold`; usa `extra="forbid"`. La respuesta
expone NMS y `max_det` efectivos y declara el comparador legacy
`strictly_greater_than`.

### 4.4 Frontera COCOeval

`ml/metrics/coco_detection.py` acepta y devuelve únicamente contratos
propios. Ningún `COCO`, `COCOeval`, array NumPy, diccionario, índice de eje,
sentinel o texto impreso cruza su frontera.

El adaptador:

- ordena entradas de forma determinista;
- asigna IDs efímeros solo dentro de COCOeval;
- conserva `category_internal_id`, ID COCO y clase del modelo fuera de esa
  frontera;
- convierte `xyxy → xywh` únicamente al crear los registros COCO;
- conserva precisión completa;
- ejecuta `evaluate()`, `accumulate()` y `summarize()`;
- redirige stdout;
- valida formas estructurales;
- rechaza NaN e infinitos;
- transforma ausencia o `-1` en `None`;
- no implementa manualmente AP, interpolación ni matching IoU.

### 4.5 Definiciones métricas

Parámetros COCO:

- IoU `0.50:0.05:0.95`;
- 101 niveles de recall;
- áreas `all`, `small`, `medium`, `large`;
- `maxDets=[1,10,100]`;
- evaluación por categoría.

Escalares agregados:

- `map_50_95`: media de precisiones válidas sobre IoU, recall y clases,
  área `all`, `maxDet=100`;
- `ap_50`, `ap_75`;
- `ar_1`, `ar_10`, `ar_100`;
- AP y AR `small`, `medium`, `large`.

Por clase:

- AP: media de precision válida sobre IoU y recall, área `all`,
  `maxDet=100`;
- average recall: media de recall válido sobre IoU, área `all`,
  `maxDet=100`;
- soporte: objetos no crowd;
- imágenes con objetos no crowd;
- número de objetos `iscrowd`;
- referencia trazable a categoría interna, categoría COCO y clase del modelo.

Las curvas usan directamente `eval["precision"]`:

- eje X: los 101 recalls COCO;
- eje Y: precision interpolada;
- área `all`, `maxDet=100`;
- media solo sobre valores válidos;
- agregado sobre IoU y clases;
- por clase sobre IoU;
- la media de sus puntos coincide con el AP correspondiente dentro de
  tolerancia numérica.

El punto de operación reutiliza `evalImgs` para un único IoU, área `all` y
`maxDet=100`. Selecciona detecciones por confidence y usa los IDs de matches
oficiales. Un match cuyo score queda bajo el threshold se vuelve FN. Respeta
`dtIgnore`, `gtIgnore` e `iscrowd`.

Política:

- `precision = TP / (TP + FP)`;
- `recall = TP / (TP + FN)`;
- `F1 = 2 × precision × recall / (precision + recall)`;
- denominador de precision cero: `null`;
- denominador de recall cero: `null`;
- precision o recall `null`: F1 `null`;
- suma precision + recall cero: F1 `null`;
- un cero con denominador válido permanece `0`.

### 4.6 Contrato y deltas

`DetectionEvaluationRequest` acepta `supervised_configuration` opcional con
defaults reproducibles. Las peticiones anteriores siguen siendo válidas.

`DetectionEvaluationResponse.supervised` contiene:

- `available`, `reason`, `warning`;
- `annotation_coverage`;
- `configuration`;
- `original`, `adversarial`, `deltas`.

Los schemas impiden NaN/infinito y validan disponibilidad, referencias,
soporte, rangos y conteos. La frontera pública añade estas invariantes:

- `images_with_objects <= support`;
- soporte cero implica AP, AR y curva PR `null`, aunque existan objetos
  `iscrowd`;
- soporte positivo implica AP, AR y curva disponibles;
- referencias de categoría y modelo no vacías, con IDs enteros estrictos;
- precision, recall y F1 deben coincidir con TP, FP y FN dentro de tolerancia
  absoluta `1e-6`, adecuada al redondeo público a seis decimales;
- `TP=FP=FN=0` implica precision, recall y F1 `null`;
- un F1 indefinido nunca se convierte en cero.

Para AP, AR, precision, recall y F1:

```text
absolute = original - adversarial
relative = (original - adversarial) / abs(original)
```

Los deltas se calculan con precisión completa y se redondean a seis decimales
solo al construir el schema. Si un operando es `null`, ambos deltas son
`null`; si el baseline es cero, el relativo es `null`.

No se generan deltas punto a punto para curvas.

### 4.7 Integración y preservación legacy

`evaluate_detection_dataset()`:

- obtiene el dataset, modelo, ataque y snapshot una vez;
- preserva la precedencia de los errores de compatibilidad de modelo/ataque;
- decide `no_annotation_snapshot` y `no_reviewed_images` sin catálogo;
- obtiene el catálogo y resuelve categorías una sola vez únicamente cuando
  hay snapshot activo e imágenes revisadas;
- aplica el ataque una vez por imagen;
- cuando las métricas están disponibles, ejecuta una inferencia científica
  original y una adversaria, conserva precisión completa y deriva el
  contrato legacy mediante `build_legacy_detection_result()`;
- esa adaptación conserva orden y cajas y usa exactamente
  `confidence > 0.25`; el score `0.25` queda excluido;
- cuando no están disponibles, usa `detect()` y nunca llama a
  `detect_for_evaluation()`;
- una indisponibilidad descubierta tras catálogo/resolución también vuelve al
  camino `detect()`;
- incluye solo imágenes revisadas en COCOeval y todas las imágenes en el
  resultado legacy;
- no repite inferencia por métrica, IoU, tamaño, clase o curva;
- no modifica `attack_success` ni `attack_success_rate`;
- valida la geometría predicha frente al snapshot y propaga un error interno
  tipado si no coincide.

## 5. Pruebas añadidas o ampliadas

### 5.1 Núcleo COCO

Nombres exactos en `tests/test_coco_detection_metrics.py`:

- `test_coco_metrics_perfect_detection`
- `test_coco_metrics_empty_predictions_are_valid_and_do_not_print`
- `test_coco_metrics_completely_incorrect_prediction`
- `test_coco_metrics_iou_can_separate_ap50_from_ap75`
- `test_coco_metrics_respect_score_ranking_in_precision_recall_curve`
- `test_coco_metrics_multiclass_ids_are_independent_from_coco_ids`
- `test_coco_metrics_include_resolved_class_with_zero_support_as_null`
- `test_coco_metrics_report_small_medium_and_large_when_supported`
- `test_coco_metrics_crowd_objects_and_matching_detections_are_ignored`
- `test_coco_metrics_input_order_does_not_change_output`
- `test_coco_metrics_ar_respects_standard_max_detections`
- `test_coco_metrics_operating_point_uses_configured_confidence`
- `test_coco_metrics_operating_point_treats_low_score_match_as_false_negative`
- `test_coco_metrics_xyxy_to_xywh_preserves_full_precision`
- `test_coco_metrics_never_expose_nan_infinity_or_sentinel`

### 5.2 Snapshot y contrato

Nombres exactos:

- `test_snapshot_reader_filters_every_collection_by_dataset_and_snapshot`
- `test_snapshot_reader_retries_without_mixing_replaced_snapshots`
- `test_snapshot_reader_raises_typed_conflict_after_bounded_retries`
- `test_snapshot_reader_without_active_snapshot_skips_annotation_queries`
- `test_materialized_snapshot_is_immutable`
- `test_snapshot_reader_does_not_read_original_coco_json`
- `test_snapshot_reader_wraps_missing_persisted_fields`
- `test_snapshot_reader_rejects_invalid_persisted_field_types`
- `test_snapshot_reader_rejects_incoherent_persisted_references`
- `test_snapshot_reader_wraps_database_errors_with_cause`
- `test_old_detection_evaluation_request_gets_reproducible_defaults`
- `test_supervised_configuration_rejects_invalid_thresholds`
- `test_operating_point_contract_accepts_rounded_consistent_metrics`
- `test_operating_point_contract_rejects_incoherent_metrics`
- `test_per_class_contract_allows_zero_support_with_crowd_objects`
- `test_per_class_contract_rejects_incoherent_support`
- `test_per_class_contract_accepts_positive_support_with_complete_metrics`
- `test_available_supervised_metrics_include_partial_coverage_and_deltas`
- `test_supervised_unavailability_is_typed`
- `test_unknown_ambiguous_and_conflicting_categories_are_counted`
- `test_relative_delta_is_null_when_original_metric_is_zero`
- `test_reviewed_empty_image_participates_as_a_negative_example`
- `test_supervised_contract_serializes_without_non_finite_values`

Las pruebas parametrizadas cubren configuración extra/thresholds inválidos,
campos ausentes, tipos inválidos, referencias incoherentes, estados
matemáticos inválidos y los cuatro motivos de indisponibilidad.

### 5.3 Integración, HTTP y YOLO

Nombres exactos añadidos o ampliados:

- `test_detection_evaluation_uses_canonical_deltas_and_legacy_success`
- `test_detection_evaluation_integrates_supervised_metrics_once`
- `test_detection_evaluation_propagates_supervised_geometry_error`
- `test_legacy_detection_result_uses_strict_historical_threshold`
- `test_detection_evaluation_skips_catalog_and_scientific_inference_without_reviewed_images`
- `test_detection_evaluation_uses_legacy_after_required_resolution_is_unavailable`
- `test_allowed_supervised_configuration_does_not_change_legacy_contract`
- `test_detect_evaluate_returns_detection_evaluation_response`
- `test_detect_evaluate_rejects_invalid_supervised_thresholds`
- `test_yolo_evaluation_inference_uses_explicit_scientific_configuration`
- `test_yolo_normal_detection_keeps_provider_defaults`
- `test_detection_evaluation_preserves_unexpected_error`

Se usan dobles y ejemplos sintéticos; no se cargan pesos ni se ejecuta
inferencia real.

Las pruebas COCO existentes se ampliaron para comprobar explícitamente curva
agregada, curva por clase, punto de operación agregado y por clase,
coherencia curva/AP, soporte cero, soporte exclusivamente crowd y ausencia de
sentinels/no finitos.

## 6. Archivos

### Nuevos

- `backend/app/ml/metrics/coco_detection.py`
- `backend/app/ml/models/detection_inference.py`
- `backend/app/services/detection_annotation_snapshot.py`
- `backend/app/services/detection_supervised_evaluation.py`
- `backend/tests/test_coco_detection_metrics.py`
- `backend/tests/test_detection_annotation_snapshot_reader.py`
- `backend/tests/test_detection_supervised_metrics_contract.py`
- `codex-reports/05-feat-backend-coco-supervised-detection-metrics.md`

### Modificados

- `backend/app/ml/models/base.py`
- `backend/app/ml/models/yolo/model.py`
- `backend/app/schemas/detection_evaluation.py`
- `backend/app/schemas/detection_metrics.py`
- `backend/app/services/detection_evaluation_service.py`
- `backend/requirements.txt`
- `backend/tests/test_detection_evaluation.py`
- `backend/tests/test_detection_evaluation_service.py`
- `backend/tests/test_unexpected_service_errors.py`
- `backend/tests/test_yolo_detection_adapter.py`

## 7. Construcción y validación de la primera entrega

Como se modificó `requirements.txt`, se ejecutó:

```text
docker compose build backend
docker compose up -d --force-recreate backend
```

El primer intento de build quedó bloqueado por las restricciones de
`snap-confine`; se repitió con la autorización correspondiente. Resultado
final: código 0, imagen backend reconstruida con `pycocotools==2.0.11` y
contenedor recreado correctamente.

### 7.1 Regresión focalizada obligatoria

Comando literal:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_coco_detection_metrics.py \
  tests/test_detection_supervised_metrics_contract.py \
  tests/test_detection_annotation_snapshot_reader.py \
  tests/test_detection_evaluation_service.py \
  tests/test_detection_evaluation.py \
  tests/test_detection_metrics.py \
  tests/test_detection_category_resolution.py \
  tests/test_detection_class_catalog.py \
  tests/test_detection_prediction_contract.py \
  tests/test_yolo_detection_adapter.py \
  tests/test_models.py \
  tests/test_coco_annotation_importer.py \
  tests/test_dataset_coco_annotations.py
```

Resultado de esa entrega:

```text
código: 0
recopiladas: 203
pasadas: 203
fallos: 0
skips: 0
warnings informados: 0
tiempo: 14.84s
```

### 7.2 Suite completa

Comando:

```text
docker compose exec backend python -m pytest -ra -W always
```

Una primera ejecución detectó tres regresiones de precedencia de errores:

```text
532 recopiladas
527 pasadas
3 fallos
2 skips
49.07s
```

Se corrigió el orden para validar modelo y ataque antes de leer el snapshot y
se adaptó el doble del test de error inesperado al nuevo colaborador.

Resultado de esa entrega:

```text
código: 0
recopiladas: 532
pasadas: 530
fallos: 0
skips: 2
warnings informados: 0
tiempo: 49.70s
```

Skips:

- `test_attack_source_geometry_reconstruction.py:148`: CUDA no disponible.
- `test_gradcam_source_geometry.py:240`: CUDA no disponible.

### 7.3 Calidad

```text
docker compose exec backend python -m ruff check app tests
```

Código 0: `All checks passed!`

```text
docker compose exec backend python -m ruff check \
  --select E4,E7,E9,F,I app tests
```

Código 0: `All checks passed!`

```text
docker compose exec backend python -m ruff format --check app tests
```

Código 0: `125 files already formatted`.

```text
docker compose exec backend python -m compileall -q app tests
```

Código 0, sin salida.

```text
git diff --check
```

Código 0, sin salida.

Ruff había señalado inicialmente tres bloques de imports y doce archivos por
formatear. Se aplicó `ruff format` y el autoarreglo limitado a imports; todas
las comprobaciones anteriores se repitieron después sobre el estado final.

## 8. Validación de la ronda correctiva

### 8.1 Focalización ampliada

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_coco_detection_metrics.py \
  tests/test_detection_supervised_metrics_contract.py \
  tests/test_detection_annotation_snapshot_reader.py \
  tests/test_detection_evaluation_service.py \
  tests/test_detection_evaluation.py \
  tests/test_yolo_detection_adapter.py \
  tests/test_unexpected_service_errors.py
```

Resultado final: código 0, 105 recopiladas, 105 pasadas, 0 fallos, 0 skips,
0 warnings informados, `2.68s`.

### 8.2 Regresión obligatoria

Se ejecutó literalmente la lista de trece módulos documentada en 7.1.

Resultado final: código 0, 234 recopiladas, 234 pasadas, 0 fallos, 0 skips,
0 warnings informados, `10.07s`.

### 8.3 Suite completa

```text
docker compose exec backend python -m pytest -ra -W always
```

Resultado final:

```text
código: 0
recopiladas: 563
pasadas: 561
fallos: 0
skips: 2
warnings informados: 0
tiempo: 28.39s
```

Skips:

- `test_attack_source_geometry_reconstruction.py:148`: CUDA no disponible.
- `test_gradcam_source_geometry.py:240`: CUDA no disponible.

### 8.4 Calidad

```text
docker compose exec backend python -m ruff check app tests
```

Código 0: `All checks passed!`

```text
docker compose exec backend python -m ruff check \
  --select E4,E7,E9,F,I app tests
```

Código 0: `All checks passed!`

```text
docker compose exec backend python -m ruff format --check app tests
```

Código 0: `125 files already formatted`.

```text
docker compose exec backend python -m compileall -q app tests
```

Código 0, sin salida.

```text
git diff --check
```

Código 0, sin salida.

## 9. Diff final, alcance y riesgos

El diff final completo se revisó mediante:

```text
git status --short
git diff --stat
git diff
git ls-files --others --exclude-standard
```

El stat de archivos tracked informa diez archivos, 1182 inserciones y 8
eliminaciones. `git ls-files --others --exclude-standard` enumera ocho
archivos nuevos, detallados en la sección 6.

El alcance queda limitado al backend, sus pruebas, `requirements.txt` y este
informe. No se modificó ningún archivo de frontend. No se leyó el JSON COCO
original, no se cambió `Detection`, `DetectionBox` ni `DetectionResult`, no se
añadió matching manual, no se persistieron métricas y no se realizó commit ni
push.

`attack_success` y `attack_success_rate` conservan su cálculo anterior. En
supervisión disponible reciben el mismo universo histórico producido por NMS
0.7, `max_det=300` y corte estricto `confidence > 0.25`; en supervisión no
disponible usan directamente `detect()`.

Riesgos pendientes:

- la equivalencia legacy depende de los defaults verificados de Ultralytics
  8.4.104; una actualización debe revisar las constantes 0.7/300 y las
  pruebas de frontera;
- COCOeval recibe la salida post-NMS del detector, no anchors crudos, que es
  la semántica estándar de evaluación de este modelo;
- no hay caché global; una evaluación supervisada disponible conserva el
  coste de una detección original y una adversaria por imagen;
- `supervised` sigue siendo opcional en el schema de respuesta por
  retrocompatibilidad, aunque el servicio real siempre lo rellena.
