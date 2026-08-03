# Informe: análisis adversario por objeto para detección

## 1. Resultado y alcance

- Estado: implementado exclusivamente en backend.
- `HEAD` inicial: `350f68a059d1335ca5858d8a4f6c7601fb323515`.
- Commit base: `350f68a feat(metrics): se han implementado métricas COCO supervisadas`.
- Estado inicial: árbol limpio respecto de ese commit.
- No se añadieron dependencias, migraciones, persistencia ni cachés.
- No se modificaron frontend, `/detect`, autenticación, autorización ni
  ownership.
- No se hizo commit ni push.

El contrato supervisado disponible incorpora
`supervised.object_analysis`. Si la supervisión no está disponible, el campo
es `null` y el motivo tipado continúa perteneciendo al bloque padre.

## 2. Arquitectura encontrada y solución

La evaluación existente ya garantizaba una inferencia original y una
adversaria por imagen, seguidas de una ejecución de `COCOeval` por condición.
El punto de operación se extraía de `evalImgs`, pero el adaptador descartaba
la relación entre el ID GT efímero de COCO y el `_id` persistido del objeto.

La solución conserva ese `_id` en `CanonicalObjectAnnotation`, lo propaga a
`CocoGroundTruth` y lo incluye como dato auxiliar en el COCO construido en
memoria. `GroundTruthMatchObservation`, dataclass propia, inmutable y sin
objetos de pycocotools, expone:

- ID interno persistido, imagen y referencia de categoría;
- estado matched;
- confidence, IoU oficial y caja de la detección matched, o tres nulos.

`_extract_operating_point()` sustituye los recorridos repetidos anteriores.
En una única pasada por el bloque `area=all` obtiene simultáneamente TP/FP/FN
globales y por clase, y una observación por GT no ignorado. Mantiene la
semántica existente `score >= confidence_threshold`; un match oficial bajo
el umbral queda unmatched en el punto de operación.

El IoU no se recalcula. `_official_iou_axes()` reconstruye y verifica una vez
por `(image_id, category_id)` los ejes reales usados por COCOeval:

- filas: detecciones ordenadas establemente por score y limitadas a maxDet;
- columnas: GT en el orden interno de COCOeval;
- shape exacta de `evaluator.ious[(image_id, category_id)]`.

Después `_official_match_iou()` resuelve fila y columna por los IDs oficiales.
Los mapas se precalculan para evitar búsquedas cuadráticas.

El nuevo servicio puro
`detection_object_adversarial_analysis.py` no conoce pycocotools ni ejecuta
matching. Valida el universo y ambas colecciones, compara por ID interno,
clasifica, agrupa y construye el contrato público. Usa diccionarios y una
única agrupación por clase, tamaño e imagen: coste O(G + C + I), además de la
ordenación determinista de salida O(G log G).

### Corrección contractual posterior

Una revisión posterior localizó tres fronteras incompletas y un artefacto:

- la identidad era opcional después de materializar el snapshot;
- los umbrales de un match solo se revalidaban al construir detalles;
- tipos numéricos incorrectos podían escapar como `TypeError`;
- `backend/detection-metrics-schema-fix.patch` seguía versionado aunque su
  contenido ya estaba aplicado.

La corrección mantiene `CanonicalObjectAnnotation.object_internal_id`
opcional únicamente para el importador antes de persistir. La preparación
supervisada exige un string no vacío mediante
`DetectionGroundTruthIdentityError`; `CocoGroundTruth` lo exige después como
campo no opcional. Se eliminaron todos los fallbacks a IDs COCO o efímeros.

El servicio puro valida siempre, antes de clasificar, tipos y rangos de GT,
observaciones, IDs, área, thresholds e `include_details`. Un match exige
`confidence >= confidence_threshold` e `IoU >= iou_threshold` tanto en modo
compacto como detallado. Todos los fallos estructurales se convierten en
`DetectionObjectAdversarialAnalysisInputError`.

## 3. Definiciones científicas

El método es estrictamente category-aware: cada observación procede del
matching oficial de COCOeval para la categoría resuelta del GT. No hay
matching predicción-original contra predicción-adversaria, matching
class-agnostic ni inferencias de cambio de clase.

Outcomes por objeto:

- `preserved`: matched original y matched adversario;
- `suppressed`: matched solo en original;
- `recovered`: matched solo en adversaria;
- `missed_both`: unmatched en ambas.

Solo para `preserved`:

```text
confidence_drop = original_confidence - adversarial_confidence
iou_drop        = original_iou - adversarial_iou
```

Los demás outcomes usan deltas nulos. Un valor negativo se conserva. Las
medias sin preserved y todas las tasas con denominador cero son nulas; un
cero definido permanece cero. El redondeo público se aplica a seis
decimales, después del cálculo a precisión completa.

Universo incluido:

- imágenes revisadas, incluidas `reviewed_empty` como negativos COCO;
- categorías resueltas;
- GT con geometría válida, `iscrowd=0` y no ignorados oficialmente.

Universo excluido:

- imágenes no revisadas;
- categorías unresolved, ambiguous o conflict;
- crowd y cualquier GT ya excluido del universo supervisado.

Las áreas usan exactamente: small `<32²`, medium `[32²,96²)` y large
`>=96²`.

## 4. Contratos e invariantes

`DetectionSupervisedConfiguration` añade el booleano estricto y
retrocompatible `include_object_level_details=false`.

El bloque público usa
`methodology_version="coco_category_aware_gt_transitions_v1"` e incluye:

- configuración del punto (`confidence`, `IoU`, comparación inclusiva y
  maxDet COCO 100);
- `summary`, `by_class` y `by_size` siempre;
- todas las clases resueltas, también con soporte cero;
- `by_image` y `objects` solo si se solicitan detalles;
- todas las imágenes revisadas en `by_image`, también las vacías;
- cajas públicas siempre `xyxy` mediante `x1,y1,x2,y2`.

Los validadores públicos comprueban:

- invariantes de outcome y presencia de observaciones/deltas;
- geometría, finitud, rangos de confidence/IoU y bucket de área;
- suma de outcomes, conteos derivados, tasas y medias;
- sumas globales por clase, tamaño e imagen;
- IDs únicos, referencias esperadas y orden canónico;
- correspondencia de objetos detallados con todos los agregados;
- coincidencia de `total_evaluable_objects` con el universo;
- ausencia de matches públicos bajo el punto declarado.

El modo compacto valida las mismas entradas y calcula las mismas transiciones
y agregados, pero no materializa `DetectionObjectResult` ni agregados
públicos por imagen.

## 5. Archivos y símbolos

Nuevos:

- `backend/app/services/detection_object_adversarial_analysis.py`:
  `DetectionObjectAdversarialAnalysisError`,
  `DetectionObjectAdversarialAnalysisInputError`, `_ObjectTransition`,
  `build_detection_object_adversarial_analysis`.
- `backend/tests/test_detection_object_adversarial_analysis.py`: 49 casos
  focalizados.
- este informe.

Modificados:

- `backend/app/ml/metrics/coco_detection.py`:
  `GroundTruthMatchObservation`, `_OfficialIouAxes`,
  `_official_iou_axes`, `_official_match_iou`,
  `_extract_operating_point` y observaciones en `CocoDetectionAnalysis`.
- `backend/app/services/annotation_importers/base.py`: identidad interna
  opcional en la anotación canónica.
- `backend/app/services/detection_annotation_snapshot.py`: conserva `_id`.
- `backend/app/services/detection_supervised_evaluation.py`: propaga la
  identidad e integra el análisis usando las dos evaluaciones existentes.
- `backend/app/schemas/detection_metrics.py`: configuración, schemas de
  objeto/agregados y validación cruzada.
- `backend/tests/test_detection_annotation_snapshot_reader.py`: identidad
  persistida.
- `backend/tests/test_detection_evaluation_service.py`: disponibilidad,
  modo compacto/completo, contrato legacy y número de inferencias.
- `backend/tests/test_detection_supervised_metrics_contract.py`: identidad
  persistida obligatoria y formateo mecánico.
- `backend/detection-metrics-schema-fix.patch`: eliminado; era un artefacto
  temporal versionado.

## 6. Pruebas focalizadas exactas

`pytest --collect-only` recopiló estos 49 IDs:

```text
test_object_transition_classifies_each_outcome[matched0-preserved]
test_object_transition_classifies_each_outcome[matched1-suppressed]
test_object_transition_classifies_each_outcome[matched2-recovered]
test_object_transition_classifies_each_outcome[matched3-missed_both]
test_all_four_outcomes_coexist_and_aggregate_globally
test_preserved_object_exposes_full_precision_drops_rounded_publicly
test_negative_preserved_drops_are_not_clamped
test_non_preserved_outcomes_have_null_deltas[matched0]
test_non_preserved_outcomes_have_null_deltas[matched1]
test_non_preserved_outcomes_have_null_deltas[matched2]
test_class_aggregation_includes_resolved_zero_support_class
test_size_aggregation_uses_exact_coco_boundaries
test_image_aggregation_includes_reviewed_empty_images
test_crowd_ground_truth_is_excluded_from_object_universe
test_details_flag_omits_only_public_per_object_and_image_materialization
test_match_below_operating_point_is_rejected_in_both_detail_modes[0.1-0.8-confidence-False]
test_match_below_operating_point_is_rejected_in_both_detail_modes[0.1-0.8-confidence-True]
test_match_below_operating_point_is_rejected_in_both_detail_modes[0.9-0.4-IoU-False]
test_match_below_operating_point_is_rejected_in_both_detail_modes[0.9-0.4-IoU-True]
test_compact_mode_does_not_construct_public_object_schemas
test_input_order_does_not_change_canonical_json
test_comparison_rejects_non_exact_observation_sets[missing]
test_comparison_rejects_non_exact_observation_sets[additional]
test_comparison_rejects_non_exact_observation_sets[duplicate]
test_comparison_rejects_incoherent_image_or_category_reference
test_comparison_rejects_match_field_invariants
test_comparison_wraps_wrong_numeric_types_in_typed_input_error[confidence]
test_comparison_wraps_wrong_numeric_types_in_typed_input_error[iou]
test_comparison_wraps_wrong_numeric_types_in_typed_input_error[area]
test_comparison_rejects_wrong_control_types_with_typed_error[confidence_threshold-bad]
test_comparison_rejects_wrong_control_types_with_typed_error[iou_threshold-True]
test_comparison_rejects_wrong_control_types_with_typed_error[include_details-1]
test_comparison_rejects_ground_truth_without_persisted_id_and_never_falls_back
test_schema_rejects_impossible_object_outcome_combination
test_schema_rejects_size_bucket_inconsistent_with_authoritative_area
test_aggregate_schema_rejects_incoherent_counts_and_rates[total_objects-2]
test_aggregate_schema_rejects_incoherent_counts_and_rates[originally_detected_count-0]
test_aggregate_schema_rejects_incoherent_counts_and_rates[suppression_rate-0.0]
test_analysis_schema_rejects_dimension_that_does_not_sum_globally
test_analysis_schema_rejects_match_below_declared_operating_point
test_zero_denominators_are_null_not_zero
test_adapter_exposes_persisted_id_and_official_iou_without_ephemeral_ids
test_adapter_uses_same_inclusive_confidence_boundary_as_operating_point
test_adapter_turns_official_low_score_match_into_unmatched_observation
test_adapter_uses_official_iou_threshold_for_object_match
test_adapter_extracts_observations_during_single_cocoeval
test_match_observation_is_an_immutable_project_dataclass
test_complete_analysis_serializes_to_strict_json_without_provider_objects[False]
test_complete_analysis_serializes_to_strict_json_without_provider_objects[True]
```

Las regresiones adyacentes añaden exactamente
`test_coco_metrics_reject_ground_truth_without_persisted_object_id` y
`test_preparation_rejects_snapshot_object_without_persisted_internal_id`.
También comprueban categorías no resueltas,
indisponibilidad con `object_analysis=null`, propagación de errores, una
inferencia original/adversaria por imagen, resolución única y equivalencia
literal de los campos legacy.

## 7. Comandos y resultados

Primer intento focal antes de corregir dos expectativas del propio test:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_object_adversarial_analysis.py
```

Código 1: 34 recopiladas, 32 pasadas, 2 fallos, 0 skips, 0 warnings,
`0.55s`. Se corrigieron el valor geométrico esperado del IoU sintético y una
búsqueda textual de `-1` que confundía `image-1` con el sentinel numérico.

Ejecución focal final:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_object_adversarial_analysis.py
```

Resultado final tras la corrección contractual: código 0, 49 recopiladas, 49
pasadas, 0 fallos, 0 skips, 0 warnings, `0.59s`.

El primer pase conjunto de los tres módulos contractuales recopiló 106 casos
y terminó con 62 pasados y 44 fallos por una guarda de
`reviewed_image_ids` colocada por error dentro del helper genérico. Se movió
a la frontera correcta y la repetición terminó en código 0: 106/106 pasadas
en `2.13s`.

Regresión obligatoria de catorce módulos, ejecutada literalmente como en la
sub-issue:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_object_adversarial_analysis.py \
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

Resultado final tras la corrección contractual: código 0, 301 recopiladas,
301 pasadas, 0 fallos, 0 skips, 0 warnings, `12.40s`.

Suite completa:

```text
docker compose exec backend python -m pytest -ra -W always
```

Resultado final tras la corrección contractual: código 0, 630 recopiladas,
628 pasadas, 0 fallos, 2 skips, 0 warnings, `29.21s`.

Skips:

- `test_attack_source_geometry_reconstruction.py:148`: CUDA no disponible.
- `test_gradcam_source_geometry.py:240`: CUDA no disponible.

Calidad:

```text
docker compose exec backend python -m ruff check app tests
# código 0: All checks passed!

docker compose exec backend python -m ruff check \
  --select E4,E7,E9,F,I app tests
# código 0: All checks passed!

docker compose exec backend python -m ruff format --check app tests
# código 0: 127 files already formatted

docker compose exec backend python -m compileall -q app tests
# código 0, sin salida

git diff --check
# código 0, sin salida
```

El primer `ruff format --check` señaló seis archivos. Se aplicó el formateador
mecánico exclusivamente a ellos y después se repitieron los chequeos. Una
regresión adicional tras la optimización de complejidad ejecutó 62 pruebas:
`62 passed in 0.84s`.

Después del único formateo mecánico de la ronda contractual se repitieron los
cuatro módulos directamente afectados: código 0, `115 passed in 1.76s`.

## 8. Diff final y confirmaciones

El diff final completo se revisó mediante:

```text
git status --short
git diff --stat
git diff
git ls-files --others --exclude-standard
```

Los únicos archivos nuevos son el servicio, las 49 pruebas y este informe.
Además se eliminó el parche temporal versionado. No hay cambios en
dependencias ni frontend.

Confirmaciones expresas:

- no se leyó el JSON COCO original;
- no se implementó IoU, AP/mAP ni matching manual;
- no se emparejaron predicciones originales y adversarias;
- no se implementaron ni afirmaron cambios de clase;
- se conservó matching category-aware de COCOeval;
- siguen existiendo exactamente una inferencia original y una adversaria por
  imagen, y una ejecución COCOeval original y otra adversaria;
- `include_object_level_details` no cambia matches, agregados ni campos
  legacy;
- no cambió el filtrado legacy, NMS 0.7 ni max_det 300;
- no se modificó frontend;
- no se hizo commit ni push.

Riesgos/limitaciones pendientes: el contrato depende, igual que las métricas
COCO existentes, de la estructura documentada de `evalImgs`, `ious`, `_dts`
y `_gts` de la versión instalada de pycocotools. La verificación explícita de
shape, ejes, IDs y finitud convierte una incompatibilidad futura en error
interno tipado en vez de producir resultados silenciosamente incorrectos.
