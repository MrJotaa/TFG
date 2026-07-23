# Informe: catálogo y resolución canónica de categorías de detección

## 1. Estado final

- Resultado: completado.
- `HEAD` inicial y final sin commit:
  `8381ea4916baac4495b0a45f9ce19eba9c5ebef8`.
- El árbol de trabajo estaba limpio al iniciar la sub-issue.
- Las sub-issues de métricas, snapshots COCO y contrato canónico estaban ya
  integradas en `HEAD`; no se revirtieron ni rediseñaron.
- No se añadieron dependencias, endpoints, schemas HTTP, persistencia ni
  cambios frontend.
- La ronda correctiva comenzó con los siete cambios sin commit de la primera
  entrega. Todos se conservaron.
- `backend/pyproject.toml` se monta ahora en `/app/pyproject.toml:ro` para que
  Ruff use dentro del contenedor la selección y clasificación de imports real
  del repositorio. No se modificó el Dockerfile.
- No se hizo commit ni push.

La frontera resultante es:

```text
YOLO.model.names
        ↓
build_ultralytics_class_catalog
        ↓
DetectionClassCatalog
        │
        ├──────────────┐
        ↓              ↓
índice por nombre   CanonicalCategory del snapshot
        └──────┬───────┘
               ↓
DetectionCategoryResolution
               ↓
category_internal_id → model class_id
```

### Defectos corregidos en la segunda ronda

1. El resolvedor convertía y ordenaba candidatos por categoría. Con `C`
   categorías y `M` clases colisionadas retenía hasta `C × M` referencias.
2. `DetectionCategoryResolution` aceptaba construcciones manuales donde dos
   matches resueltos reclamaban la misma clase y no verificaba completamente
   la coherencia nominal con el catálogo.
3. `YOLODetectionModel.get_detection_class_catalog()` dejaba escapar
   `AttributeError` u otros errores ordinarios del proveedor al leer
   `self.model.names`.

La corrección preservó la política funcional, el contrato de predicciones y
la separación entre IDs COCO y del detector.

## 2. Estado y arquitectura inicial

### Modelos y predicciones

`Detection`, `DetectionBox` y `DetectionResult` ya proporcionaban la
predicción agnóstica. Cada detección conservaba:

- `class_id`, procedente de `Results.boxes.cls`;
- `class_name`, resuelto mediante `Results.names`;
- confianza completa;
- caja `xyxy` en geometría fuente.

`BaseObjectDetectionModel` no exponía el catálogo completo. Solo una
inferencia podía revelar las clases presentes en sus resultados, lo que no es
suficiente para resolver categorías ausentes de una imagen.

### Anotaciones

`CanonicalCategory`, producido por el importador COCO, conserva:

- `internal_id`: identidad propia del snapshot usada por objetos persistidos;
- `external_id`: `categories[].id` del JSON COCO;
- `name`;
- `supercategory`.

En MongoDB las categorías se almacenan con `dataset_id`, `snapshot_id`,
`coco_category_id`, nombre e ID interno. Los objetos apuntan al ID interno. El
servicio de cobertura ya filtra categorías y objetos mediante
`dataset_id + active_annotation_snapshot_id`.

Esta sub-issue no añadió otra lectura de persistencia ni conectó el resolvedor
con `evaluate_detection_dataset`.

### Precedente de clasificación

`classification_label_service.py` dispone de normalización, aliases ImageNet,
eliminación de puntuación y respuestas HTTP específicas. No se reutilizó
directamente porque esas reglas y contratos pertenecen a clasificación.
Extraerlos habría ampliado el alcance y podría alterar su comportamiento.

## 3. Dos espacios de identidad

Los IDs no son comparables:

| Espacio | ID | Significado |
|---|---|---|
| Snapshot | `category.internal_id` | Relación interna categoría–objetos |
| COCO | `category.external_id` | `categories[].id` referenciado por `annotation.category_id` |
| Detector | `Detection.class_id` | Índice usado para consultar el catálogo del modelo |

Por ejemplo, una categoría COCO `{id: 1, name: "person"}` puede resolverse a
`DetectionClass(class_id=0, name="person")`. Comparar `1 == 0` fallaría; usar el
ID COCO 1 como índice del modelo podría seleccionar otra clase.

El resolvedor solo usa el nombre normalizado. Ambos IDs del dataset se
conservan exclusivamente para trazabilidad y para enlazar objetos después de
resolver.

## 4. Fuentes primarias consultadas

Fecha de consulta: **2026-07-23**.

### Ultralytics

Versión real instalada:

```text
Name: ultralytics
Version: 8.4.104
Location: /usr/local/lib/python3.10/site-packages
```

Se inspeccionó con `inspect.getsource` el código instalado de
`Model.names.fget` y `check_class_names`:

- `Model.names` devuelve `dict[int, str]` desde `self.model.names`;
- `check_class_names` admite `list | dict`;
- una lista se convierte mediante `dict(enumerate(names))`;
- las claves de un diccionario se convierten a enteros.

Fuentes oficiales:

- [Ultralytics `Model.names`](https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.names):
  documenta el mapping índice–nombre y la fuente `model.names`.
- [Ultralytics Detect / Predict](https://docs.ultralytics.com/tasks/detect/#predict):
  consulta `result.names[cls.item()]`, confirmando que `boxes.cls` usa las
  claves del catálogo.
- [Ultralytics `Results` y `Boxes.cls`](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes.cls):
  define `cls` como IDs de clase de las cajas.
- [Código oficial de la versión 8.4.104](https://github.com/ultralytics/ultralytics/blob/v8.4.104/ultralytics/engine/model.py):
  referencia versionada del proveedor inspeccionado localmente.

Decisión respaldada: `YOLODetectionModel` usa `self.model.names` sin forward.
Los mappings preservan sus claves; una secuencia solo se enumera porque esa es
la conversión contractual del proveedor.

### COCO

Fuentes oficiales:

- [COCO API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py):
  indexa `categories` por `cat["id"]` y relaciona anotaciones mediante
  `ann["category_id"]`.
- [COCO Data Format](https://cocodataset.org/#format-data): documentación
  primaria del formato de datos.

Decisión respaldada: `category_id` identifica una categoría del dataset COCO;
no expresa la posición de una clase en un detector independiente.

## 5. Contrato del catálogo

`ml/detection_categories.py` introduce:

- `DetectionClass(class_id, name)`;
- `DetectionClassCatalog(model_id, model_name, classes)`;
- `DetectionClassNameCollision`;
- `DetectionClassCatalogError`.

Invariantes:

- ID entero y no booleano;
- ID no negativo y único;
- nombre no vacío ni vacío tras normalización;
- catálogo no vacío;
- `model_id` y `model_name` no vacíos;
- copia a tupla y orden estable por `class_id`;
- mappings no contiguos preservados;
- ningún objeto del proveedor, tensor, array o schema HTTP.

Los nombres duplicados o que colisionan tras normalización no se eligen
silenciosamente. El catálogo los expone en
`normalized_name_collisions`; el resolvedor los convierte en una ambigüedad
con todos los candidatos. Esta decisión permite diagnosticar un catálogo
estructuralmente representable sin inventar una clase ganadora.

Un catálogo vacío se considera estructuralmente inválido: un detector cargado
debe poder describir al menos una clase. Esto es independiente de que un
snapshot de dataset pueda no contener categorías.

## 6. Adaptación del catálogo YOLO

`build_ultralytics_class_catalog()` es la única frontera nueva que entiende
los formatos de `names`:

- mapping: usa cada clave explícita como `class_id`, sin enumerar ni exigir
  continuidad;
- secuencia: usa `enumerate`, de acuerdo con `check_class_names` de
  Ultralytics 8.4.104;
- otro formato: `UltralyticsClassCatalogError`.

`BaseObjectDetectionModel.get_detection_class_catalog()` hace explícita la
capacidad. `YOLODetectionModel` la implementa leyendo `self.model.names`. Si el
modelo aún no está cargado, lo carga, pero no llama a `_run_inference`,
`predict` ni `forward`.

Los fallos ordinarios al cargar el modelo para obtener el catálogo y al leer
la propiedad `names` se convierten en `UltralyticsClassCatalogError`. El
mensaje contiene `model_id` y `model_name`, distingue carga de lectura y la
excepción original se conserva como `__cause__` mediante `raise ... from`.
No se captura `BaseException`.

El adaptador de predicciones sigue leyendo los mismos IDs de `boxes.cls`. Una
prueba conjunta exige que una predicción con clase 4 corresponda a la clase 4
del catálogo construido desde el mismo mapping.

## 7. Normalización

`normalize_detection_category_name()` aplica exactamente:

1. validación de cadena no vacía;
2. Unicode NFKC;
3. `casefold`;
4. guion ASCII `-` y guion bajo `_` como espacios;
5. colapso de espacios repetidos.

No elimina otra puntuación, no separa aliases, no traduce, no aplica stemming
y no realiza fuzzy matching.

Tratar guiones y guiones bajos como espacios permite resolver diferencias
sintácticas como `fire-truck`, `fire_truck` y `Fire Truck`. Las colisiones que
esta regla produzca son explícitas y se prueban.

## 8. Resolución y trazabilidad

`resolve_detection_categories()` recibe:

- `DetectionClassCatalog`;
- una secuencia estructuralmente tipada de categorías canónicas con
  `internal_id`, `external_id` y `name`.

No recibe MongoDB, JSON COCO, Pydantic, FastAPI ni Ultralytics.

El resultado:

- `DetectionCategoryResolution`;
- `DetectionCategoryMatch` por categoría;
- estado `resolved`, `unresolved`, `ambiguous` o `conflict`;
- motivo tipado cuando no se resuelve;
- candidatos ordenados para ambigüedad/conflicto;
- nombre original y normalizado;
- ID interno y COCO;
- catálogo/modelo de origen;
- totales, cobertura y conteos;
- `resolved_category_class_pairs`, una tupla inmutable;
- `class_id_by_category_internal_id`, un diccionario nuevo por acceso.

Política:

- cero candidatos: categoría desconocida, no excepción;
- varios candidatos del modelo: ambigüedad;
- una clase reclamada por varias categorías del dataset: todas quedan en
  conflicto, sin ganador dependiente del orden;
- una categoría y una clase: resuelta;
- clases del modelo ausentes del dataset: no reducen cobertura;
- snapshot sin categorías: resultado vacío y cobertura `0.0`.

La correspondencia reutilizable se construye una vez por resolución y evita
buscar la clase para cada objeto.

## 9. Determinismo, complejidad e inmutabilidad

El índice local se materializa como:

```text
MappingProxyType[
    nombre_normalizado,
    tuple[DetectionClass, ...]
]
```

Cada grupo se congela una sola vez. Como el catálogo ya está ordenado por
`class_id`, cada tupla de candidatos queda ordenada determinísticamente. Todos
los matches ambiguos con el mismo nombre almacenan una referencia a esa misma
tupla; no se reconstruye ni se copia por categoría.

- Construcción del índice nominal: `O(M)`.
- Consultas del índice para todas las categorías: `O(C)`.
- Construcción más consulta del índice: `O(M + C)`.
- Ordenación del catálogo al construirlo: `O(M log M)`.
- Preparación y ordenación final de categorías/matches: `O(C log C)`.
- Validación de contratos: `O(C + R)`, donde `R` es el número de referencias
  candidatas inspeccionadas. `DetectionCategoryResolution` valida cada tupla
  compartida una vez por nombre; la construcción pública individual de cada
  `DetectionCategoryMatch` sigue validando sus candidatos.
- Memoria adicional retenida: `O(M + C)`. En el caso patológico de `C`
  categorías ambiguas y `M` clases colisionadas se retienen `M` referencias
  dentro de una sola tupla y `C` referencias a esa tupla, no `C × M`
  referencias.

No existe búsqueda lineal por todo el catálogo para cada categoría u objeto.
La futura evaluación deberá resolver una vez por pareja
`snapshot_id–catálogo`, no por imagen.

Todos los contratos usan dataclasses congeladas con `slots` y tuplas. Las
propiedades que exponen diccionarios generan una copia; no almacenan
colecciones mutables compartidas.

`DetectionCategoryResolution.__post_init__()` rechaza:

- IDs internos de categoría duplicados;
- dos matches `RESOLVED` que reclamen el mismo `class_id`;
- una clase resuelta que no pertenezca al catálogo;
- una clase resuelta cuyo nombre no coincida con el nombre normalizado del
  match;
- candidatos ausentes del catálogo;
- candidatos cuyo nombre normalizado no coincida;
- IDs candidatos duplicados;
- combinaciones incoherentes de estado, motivo, clase resuelta y candidatos.

Estas excepciones solo afectan a construcciones manuales inválidas. El
resolvedor sigue representando desconocidos como `UNRESOLVED`, colisiones
nominales como `AMBIGUOUS` y reclamaciones múltiples como `CONFLICT`.

No se añadió caché global porque el modelo activo y el snapshot pueden cambiar
concurrentemente.

## 10. Snapshots y persistencia

No se añadió un helper de lectura porque esta sub-issue no integra aún la
evaluación. Las futuras métricas deberán:

1. fijar `active_annotation_snapshot_id`;
2. leer categorías y objetos con `dataset_id + snapshot_id`;
3. convertir las categorías persistidas a `CanonicalCategory`;
4. resolver una vez el catálogo;
5. reutilizar `category_internal_id → class_id`.

No debe consultarse el JSON original ni mezclar documentos de snapshots.

## 11. Archivos y símbolos

Cambios funcionales:

- `backend/app/ml/models/base.py`
  - `BaseObjectDetectionModel.get_detection_class_catalog`.
- `backend/app/ml/models/yolo/model.py`
  - `YOLODetectionModel.get_detection_class_catalog`;
  - tipado y encadenamiento de fallos de carga/lectura de `names`.
- `backend/app/ml/detection_categories.py`
  - contratos, normalizador, estados y resolvedor;
  - `_build_classes_by_normalized_name`;
  - reutilización de candidatos;
  - invariantes de `DetectionCategoryResolution`.
- `backend/app/ml/models/yolo/catalog.py`
  - `UltralyticsClassCatalogError`;
  - `build_ultralytics_class_catalog`.
- `backend/tests/test_detection_class_catalog.py`.
- `backend/tests/test_detection_category_resolution.py`.
- `backend/docker-compose.yml`
  - montaje de solo lectura de `pyproject.toml`.
- `codex-reports/04-feat-backend-detection-category-resolution.md`.

La selección explícita de Ruff encontró 36 bloques de imports desordenados.
Su autofix `I` modificó únicamente imports en 36 archivos backend; uno era
`ml/models/yolo/model.py` y los otros 35 no recibieron cambios de lógica. El
estado completo se registra en la sección 15.

No se cambió el comportamiento del importador COCO, servicio de snapshots,
`detection_evaluation_service`, schemas, rutas, métricas o clasificación. No
se modificaron frontend ni dependencias.

## 12. Pruebas añadidas

### `test_detection_class_catalog.py`

1. `test_ultralytics_class_catalog_preserves_explicit_mapping_ids`
2. `test_ultralytics_class_catalog_enumerates_sequence_ids`
3. `test_detection_class_rejects_invalid_id`
4. `test_detection_class_catalog_rejects_duplicate_ids`
5. `test_detection_class_rejects_empty_name`
6. `test_detection_class_catalog_detects_normalized_name_collisions`
7. `test_detection_class_catalog_rejects_empty_catalog`
8. `test_yolo_model_gets_catalog_without_running_forward`
9. `test_yolo_model_wraps_missing_names_as_typed_catalog_error`
10. `test_yolo_model_wraps_provider_names_error_and_preserves_cause`
11. `test_detection_class_catalog_does_not_leak_provider_objects`
12. `test_catalog_ids_match_prediction_adapter_class_ids`
13. `test_ultralytics_class_catalog_rejects_unsupported_names_format`

Los casos parametrizados cubren ID negativo, ID booleano, duplicado exacto,
colisión normalizada y tres formatos de `names` no admitidos.

### `test_detection_category_resolution.py`

1. `test_resolution_matches_name_without_comparing_coco_and_model_ids`
2. `test_resolution_matches_arbitrary_external_id_by_name`
3. `test_resolution_normalizes_unicode_case_and_syntactic_spacing`
4. `test_resolution_reports_unknown_category_without_exception`
5. `test_resolution_does_not_use_fuzzy_matching`
6. `test_resolution_reports_normalized_model_name_ambiguity`
7. `test_resolution_reuses_one_candidate_tuple_for_equivalent_ambiguities`
8. `test_resolution_marks_all_categories_competing_for_one_class_as_conflicts`
9. `test_resolution_contract_rejects_duplicate_resolved_class_ids`
10. `test_resolution_contract_rejects_resolved_class_outside_catalog`
11. `test_resolution_contract_rejects_resolved_class_name_mismatch`
12. `test_resolution_contract_rejects_candidate_outside_catalog`
13. `test_resolution_contract_rejects_candidate_name_mismatch`
14. `test_category_match_rejects_duplicate_candidate_ids`
15. `test_resolution_contract_accepts_valid_resolver_output`
16. `test_resolution_reports_partial_coverage_and_builds_internal_id_mapping`
17. `test_resolution_accepts_dataset_without_categories`
18. `test_resolution_order_is_deterministic`
19. `test_resolution_result_is_immutable_and_does_not_share_input_collection`
20. `test_resolution_does_not_modify_prediction_contract_precision`

La primera prueba es la regresión explícita contra resolución por ID:
COCO ID 1, nombre `person`, se resuelve a modelo ID 0 aunque el modelo ID 1
existe y corresponde a `bicycle`.

Los dos módulos recopilan 39 casos: 17 en catálogo y 22 en resolución, al
contar parametrizaciones.

## 13. Comandos y resultados reales

Los resultados hasta “Calidad” pertenecen a la entrega original. La ronda
correctiva y sus validaciones definitivas se detallan a continuación.

### Estado

```text
git rev-parse HEAD
git status --short
find .. -name AGENTS.md -print
```

Resultados:

- HEAD: `8381ea4916baac4495b0a45f9ce19eba9c5ebef8`;
- estado inicial limpio;
- único `AGENTS.md` bajo `frontend/node_modules/recharts`, no aplicable.

### Versión e inspección del proveedor

```text
docker compose exec backend python -m pip show ultralytics
```

Primer intento bloqueado por `snap-confine`; repetición aprobada: código 0,
versión `8.4.104`.

```text
docker compose exec backend python -c \
  'import inspect, ultralytics; ...'
```

Resultado: código 0; confirmó versión, `Model.names` y
`check_class_names(list | dict)`.

### Línea base

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_prediction_contract.py \
  tests/test_yolo_detection_adapter.py tests/test_models.py \
  tests/test_coco_annotation_importer.py \
  tests/test_dataset_coco_annotations.py tests/test_detection_metrics.py \
  tests/test_detection_evaluation_service.py \
  tests/test_classification_label_resolution.py
```

Resultado: `128 passed in 6.22s`.

### Tests nuevos

Primera ejecución: `28 passed in 0.43s`.

Después de ampliar colisiones exactas/normalizadas:
`29 passed in 0.49s`.

### Regresión focalizada final

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_class_catalog.py \
  tests/test_detection_category_resolution.py \
  tests/test_detection_prediction_contract.py \
  tests/test_yolo_detection_adapter.py tests/test_models.py \
  tests/test_coco_annotation_importer.py \
  tests/test_dataset_coco_annotations.py tests/test_detection_metrics.py \
  tests/test_detection_evaluation_service.py \
  tests/test_classification_label_resolution.py
```

Resultado: `157 passed in 8.69s`.

### Suite completa

```text
docker compose exec backend python -m pytest -ra -W always
```

Resultado:

```text
478 collected
476 passed, 2 skipped in 24.90s
```

Omisiones:

- CUDA no disponible en
  `test_attack_source_geometry_reconstruction.py:148`;
- CUDA no disponible en `test_gradcam_source_geometry.py:240`.

### Calidad

```text
docker compose exec backend python -m ruff check app tests
docker compose exec backend python -m ruff format --check app tests
docker compose exec backend python -m compileall -q app tests
git diff --check
```

Resultados finales:

- Ruff: `All checks passed!`;
- formato: `118 files already formatted`;
- `compileall`: código 0, sin salida;
- `git diff --check`: código 0, sin salida.

En la primera entrega Ruff detectó un import `Mapping` no usado y el formato
señaló dos archivos, y después uno tras endurecer invariantes. Se corrigieron
y esas comprobaciones pasaron usando la configuración predeterminada del
contenedor, porque entonces `pyproject.toml` no estaba montado.

### Ronda correctiva: estado inicial

```text
git rev-parse HEAD
git status --short
```

Código 0. `HEAD` seguía en
`8381ea4916baac4495b0a45f9ce19eba9c5ebef8`. Existían los dos archivos
modificados y cinco archivos no rastreados de la primera entrega; no se
descartó ninguno.

El único `AGENTS.md` hallado estaba bajo
`frontend/node_modules/recharts/AGENTS.md` y no era aplicable al backend.

### Línea base focalizada de la corrección

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_class_catalog.py \
  tests/test_detection_category_resolution.py
```

Código 0: `29 collected`, `29 passed in 0.49s`, sin fallos, skips ni warnings
de pytest.

### Ruff antes de corregir su configuración

```text
docker compose exec backend python -m ruff check \
  --select E4,E7,E9,F,I app tests
```

Código 1: `Found 90 errors`, todos `I001`. Compose montaba `app`, `tests`,
`pytest.ini` y `uploads`, mientras el Dockerfile solo copiaba
`requirements.txt` y `app`. Ruff no veía ni la selección ni
`src = ["app", "tests"]` de `pyproject.toml`.

Se añadió el volumen:

```text
./pyproject.toml:/app/pyproject.toml:ro
```

Y se recreó el backend:

```text
docker compose up -d --force-recreate backend
```

Código 0. Mongo permaneció en ejecución y `tfg-backend` fue recreado,
arrancado y quedó operativo.

Con la configuración real, tanto el comando normal como el explícito
encontraron 36 `I001` genuinos. Se aplicó exclusivamente el autofix de imports:

```text
docker compose exec backend python -m ruff check --select I --fix app tests
```

Código 0: `Found 36 errors (36 fixed, 0 remaining)`.

### Tests focalizados corregidos

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_class_catalog.py \
  tests/test_detection_category_resolution.py
```

Código 0: `39 collected`, `39 passed in 0.50s`, sin fallos, skips ni warnings
de pytest.

### Regresión relacionada corregida

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_class_catalog.py \
  tests/test_detection_category_resolution.py \
  tests/test_detection_prediction_contract.py \
  tests/test_yolo_detection_adapter.py \
  tests/test_models.py \
  tests/test_coco_annotation_importer.py \
  tests/test_dataset_coco_annotations.py \
  tests/test_detection_metrics.py \
  tests/test_detection_evaluation_service.py \
  tests/test_classification_label_resolution.py
```

Código 0: `167 collected`, `167 passed in 7.27s`, sin fallos, skips ni
warnings de pytest.

### Suite completa corregida

```text
docker compose exec backend python -m pytest -ra -W always
```

Código 0:

```text
488 collected
486 passed, 2 skipped in 23.71s
```

Skips:

- `tests/test_attack_source_geometry_reconstruction.py:148`: CUDA no
  disponible;
- `tests/test_gradcam_source_geometry.py:240`: CUDA no disponible.

No hubo fallos ni warnings.

### Calidad final de la corrección

```text
docker compose exec backend python -m ruff check app tests
```

Código 0: `All checks passed!`. Al estar montado
`/app/pyproject.toml`, utilizó `target-version = "py310"`,
`line-length = 88`, `src = ["app", "tests"]` y la selección
`E4,E7,E9,F,I`.

```text
docker compose exec backend python -m ruff check \
  --select E4,E7,E9,F,I app tests
```

Código 0: `All checks passed!`. Utilizó el mismo `pyproject.toml` y reafirmó
explícitamente la selección requerida.

```text
docker compose exec backend python -m ruff format --check app tests
```

Código 0: `118 files already formatted`.

```text
docker compose exec backend python -m compileall -q app tests
```

Código 0, sin salida.

```text
git diff --check
```

Código 0, sin salida.

## 14. Validaciones no ejecutadas

- No se descargaron pesos ni se ejecutó inferencia real: el catálogo se probó
  con una instancia cargada instrumentada y `Results` real.
- No se ejecutaron los dos caminos CUDA; pytest los omitió por no haber GPU.
- No existe type checker configurado; se ejecutó `compileall`.
- No se ejecutaron tests frontend porque no hubo cambios frontend.
- No se añadió una integración Mongo del resolvedor: no existe lectura nueva y
  la conexión consistente con snapshot/objetos pertenece a la sub-issue
  posterior.

## 15. Diff final completo

`git diff --stat` para archivos rastreados:

```text
38 files changed, 106 insertions(+), 91 deletions(-)
```

De esos archivos rastreados, 35 contienen exclusivamente el orden de imports
aplicado por Ruff. Los cuatro archivos nuevos de código/tests suman 1290
líneas. El informe es el quinto archivo no rastreado.

Estado completo:

```text
 M backend/app/api/dependencies/auth.py
 M backend/app/api/routes/auth.py
 M backend/app/api/routes/datasets.py
 M backend/app/api/routes/models.py
 M backend/app/api/routes/users.py
 M backend/app/core/error_handlers.py
 M backend/app/core/task_compatibility.py
 M backend/app/main.py
 M backend/app/ml/attacks/registry.py
 M backend/app/ml/metrics/classification.py
 M backend/app/ml/models/base.py
 M backend/app/ml/models/yolo/model.py
 M backend/app/schemas/datasets.py
 M backend/app/services/annotation_importers/coco.py
 M backend/app/services/annotation_importers/registry.py
 M backend/app/services/auth_service.py
 M backend/app/services/classification_label_service.py
 M backend/app/services/dataset_annotation_service.py
 M backend/app/services/dataset_service.py
 M backend/app/services/detection_attack_service.py
 M backend/app/services/detection_evaluation_service.py
 M backend/app/services/detection_service.py
 M backend/app/services/explainability_service.py
 M backend/app/services/model_service.py
 M backend/app/services/user_service.py
 M backend/app/utils/file_manager.py
 M backend/app/utils/result_files.py
 M backend/app/utils/security.py
 M backend/docker-compose.yml
 M backend/tests/conftest.py
 M backend/tests/test_application_errors.py
 M backend/tests/test_classification_curve_metrics.py
 M backend/tests/test_classification_metrics_calculation.py
 M backend/tests/test_classification_metrics_schema.py
 M backend/tests/test_detection_evaluation_service.py
 M backend/tests/test_service_error_architecture.py
 M backend/tests/test_standardized_service_errors.py
 M backend/tests/test_task_compatibility.py
?? backend/app/ml/detection_categories.py
?? backend/app/ml/models/yolo/catalog.py
?? backend/tests/test_detection_category_resolution.py
?? backend/tests/test_detection_class_catalog.py
?? codex-reports/04-feat-backend-detection-category-resolution.md
```

La revisión manual confirmó:

- ninguna comparación entre COCO ID y `class_id`;
- ningún cambio en `DetectionResult`, `Detection` o `DetectionBox`;
- ningún acceso Mongo ni lectura de JSON en el resolvedor;
- ningún cambio en `/detect/evaluate`;
- ninguna estructura Ultralytics, tensor o array en los contratos;
- ninguna métrica supervisada o asociación espacial;
- ningún cambio frontend;
- ningún commit ni push.

## 16. Riesgos y trabajo pendiente

- `requirements.txt` no fija la versión de Ultralytics. La implementación se
  verificó con 8.4.104; una actualización futura debe volver a comprobar
  `Model.names` y `check_class_names`.
- Los nombres distintos semánticamente no se traducen ni aproximan. Quedan
  desconocidos de forma intencionada.
- Las colisiones por normalización producen ambigüedad; requieren corregir el
  catálogo o proporcionar una política explícita futura, nunca escoger por
  orden.
- La futura integración debe fijar snapshot y catálogo durante toda la
  evaluación. Una caché necesitaría modelo, versión de catálogo y snapshot.
- Falta deliberadamente el motor de asociación espacial por objetos.

## 17. Fuera de alcance confirmado

No se implementaron:

- IoU ni matching espacial;
- TP, FP o FN;
- precision, recall, AP ni mAP;
- conexión con `/detect/evaluate`;
- persistencia o caché de la resolución;
- endpoints o schemas HTTP;
- aliases COCO/YOLO hardcodeados;
- fuzzy matching, traducción o stemming.

## 18. Checklist

- Cumplido — catálogo interno completo, tipado e inmutable.
- Cumplido — operación explícita en `BaseObjectDetectionModel`.
- Cumplido — YOLO obtiene `model.names` sin forward.
- Cumplido — mapping y secuencia adaptados conforme a la versión instalada.
- Cumplido — IDs explícitos y no contiguos preservados.
- Cumplido — clases inválidas y catálogo vacío producen errores tipados.
- Cumplido — colisiones detectadas y representadas como ambigüedad.
- Cumplido — normalización NFKC/casefold/sintáctica documentada.
- Cumplido — resolución pura por nombre, nunca por ID.
- Cumplido — desconocidos, ambigüedades y conflictos explícitos.
- Cumplido — cobertura y snapshot vacío definidos.
- Cumplido — correspondencia interna reutilizable.
- Cumplido — complejidad del índice `O(C + M)` y memoria `O(C + M)`.
- Cumplido — una única tupla candidata por nombre se reutiliza por identidad.
- Cumplido — resultados manuales incoherentes se rechazan con error tipado.
- Cumplido — fallos de `model.names` se tipan y conservan en `__cause__`.
- Cumplido — predicciones, HTTP, métricas legacy y clasificación intactos.
- Cumplido — snapshots/importación sin cambios.
- Cumplido — Ruff normal y `--select E4,E7,E9,F,I` usan la configuración real.
- Cumplido — tests focalizados, suite, Ruff, formato y compilación correctos.
- Cumplido — IoU y métricas supervisadas siguen fuera de alcance.
- Cumplido — diff completo revisado.
- Cumplido — no se hizo commit ni push.
