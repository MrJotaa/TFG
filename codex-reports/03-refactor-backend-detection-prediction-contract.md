# Informe: contrato agnóstico de predicciones de detección

## 1. Estado final

- Resultado: completado.
- `HEAD` inicial y final sin commit:
  `c7aa9c4c54fc025a87b078203050eda12aaa487c`.
- El árbol de trabajo estaba limpio al comenzar esta sub-issue.
- Se conservaron los cambios previos ya integrados en `HEAD`.
- No se añadieron dependencias, endpoints, migraciones ni cambios frontend.
- No se hizo commit ni push.

La revisión posterior al refactor detectó y corrigió dos defectos acotados:

- la rama legacy de `attack_success` estaba recibiendo confianzas canónicas
  completas en vez de reproducir el redondeo individual histórico;
- un ID de clase negativo podía indexar por el final un catálogo secuencial de
  nombres antes de que el contrato lo rechazara.

La frontera resultante es:

```text
Ultralytics Results
        ↓
UltralyticsDetectionAdapter
        ↓
DetectionResult / Detection / DetectionBox
        ↓
servicios, métricas y dibujo
        ↓
to_detection_predictions
        ↓
schemas HTTP existentes
```

## 2. Comprobaciones iniciales e instrucciones

Se inspeccionó el repositorio antes de modificarlo:

- `git rev-parse HEAD` devolvió
  `c7aa9c4c54fc025a87b078203050eda12aaa487c`;
- `git status --short` no mostró cambios;
- la búsqueda de `AGENTS.md` solo encontró uno bajo
  `frontend/node_modules/recharts`, no aplicable al backend ni a este alcance;
- el baseline focalizado recopiló 81 pruebas y terminó con `81 passed`.

## 3. Arquitectura inicial

### Registro, carga e interfaces

`services/model_service.py` mantiene un registro simple de instancias:
`ResNet18Model` para clasificación y `YOLODetectionModel` para detección.
El registro selecciona, carga y descarga modelos, pero no transforma
predicciones.

`BaseObjectDetectionModel` declaraba `detect()` y
`detect_with_visualization()` con retorno `DetectionResponse`, un schema
Pydantic HTTP. La abstracción de modelo dependía por tanto de la API pública.

La clasificación ya disponía de un contrato interno separado,
`DetailedClassificationPrediction`, además de su respuesta pública. No era
necesario ni correcto introducirla en el contrato de detección.

### Formato de detección anterior

`YOLODetectionModel._result_to_response()` consumía directamente
`ultralytics.engine.results.Results` y, por cada caja:

- ejecutaba `tolist()` para `xyxy`;
- ejecutaba `item()` para confianza;
- ejecutaba `item()` para clase;
- resolvía `result.names`;
- redondeaba confianza a cuatro decimales y coordenadas a dos;
- construía `BoundingBox`, `DetectionPrediction` y `DetectionResponse`.

El resultado del modelo era ya el schema HTTP, sin dimensiones fuente ni ID de
clase. Además, la conversión GPU/CPU/Python podía repetirse por caja y se
perdía precisión antes de que métricas u otros consumidores recibieran los
datos.

### Consumidores y acoplamientos

Se localizaron estos consumidores:

- `detection_service`: inferencia principal, persistencia de imagen y metadata;
- `detection_attack_service`: comparación original/adversaria, métricas,
  explicaciones y respuesta;
- `detection_evaluation_service`: evaluación por dataset y agregados;
- `explainability_service`: detección, EigenCAM y respuesta;
- `draw_detections_on_image`: accedía a los campos HTTP `bbox` y `label`;
- `utils/detection_metrics`: ya era agnóstico mediante un protocolo mínimo que
  solo exige `confidence`.

Los tests de evaluación y errores inesperados contenían dobles que devolvían
`DetectionPrediction` o `SimpleNamespace`, reproduciendo el acoplamiento
anterior.

Las dependencias PyTorch/NumPy que siguen en Grad-CAM y en el ataque de
confianza YOLO son necesidades propias de inferencia/explicabilidad. Permanecen
encapsuladas en esas capas y no forman parte de la predicción canónica.

## 4. Contrato canónico

`ml/models/detection.py` incorpora dataclasses inmutables con `slots`:

- `DetectionBox`: `x_min`, `y_min`, `x_max`, `y_max`;
- `Detection`: `class_id`, `class_name` opcional, `confidence`, `box`;
- `DetectionResult`: `source_width`, `source_height`, tupla de detecciones;
- `DetectionContractError`: error tipado de invariantes de dominio.

El contrato contiene exclusivamente semántica común. No incluye nombres de
modelo, dispositivo, tensor, ruta, tiempos, datos de letterboxing ni metadata
de explicabilidad.

### Invariantes

- dimensiones fuente enteras y estrictamente positivas;
- ID de clase entero, no booleano y no negativo;
- nombre de clase opcional, pero cadena no vacía cuando existe;
- confianza finita e inclusiva en `[0, 1]`;
- coordenadas reales y finitas;
- coordenadas no negativas;
- orden estricto `x_min < x_max` y `y_min < y_max`;
- `x_max <= source_width` y `y_max <= source_height`;
- resultado vacío válido;
- copia de cualquier iterable de entrada a una tupla;
- objetos congelados, sin colecciones mutables compartidas.

Los escalares aceptados se normalizan a `int` o `float`, por lo que tampoco
quedan escalares de un proveedor dentro del contrato.

## 5. Coordenadas y política geométrica

La única convención interna es:

- `xyxy`;
- píxeles continuos de la imagen fuente;
- precisión completa;
- origen en la esquina superior izquierda;
- extremos mínimos inclusivos y extremos máximos permitidos hasta la dimensión
  de la imagen.

Ultralytics aplica internamente resize/letterboxing y proyecta `boxes.xyxy` de
`Results` de vuelta a `result.orig_shape`. El adaptador recibe además las
dimensiones de la imagen PIL y exige que coincidan con `orig_shape`. Así se
reutiliza la geometría autoritativa del proveedor y no se aplica una segunda
escala uniforme incorrecta.

Política en la frontera:

- caja parcialmente fuera: se recorta a `[0, width] × [0, height]`;
- caja completamente fuera: el recorte sería degenerado y se rechaza;
- caja invertida o degenerada antes del recorte: se rechaza;
- `NaN` o infinito: se rechaza antes del recorte;
- confianza o clase inválida: se rechaza;
- caja canónica fuera de límites: el núcleo siempre la rechaza.

Por tanto, la normalización específica solo ocurre en el adaptador y ningún
consumidor necesita repetirla o decidir por proveedor.

## 6. Adaptador YOLO

`UltralyticsDetectionAdapter` es el único punto que conoce `Results`,
`boxes.xyxy`, `boxes.conf`, `boxes.cls` y `result.names`.

Para `N` detecciones:

1. concatena `xyxy`, confianza y clase en un tensor `N × 6`;
2. realiza un único `detach().to(device="cpu").tolist()` por resultado;
3. valida y recorta cada fila;
4. crea valores canónicos inmutables;
5. deja que el tensor y la lista intermedia salgan de alcance al retornar.

`DetectionAdapterError`, subtipo de `DetectionContractError`, añade contexto de
fila sin modificar contratos de errores HTTP.

El modelo conserva el `Results` nativo solo dentro de
`detect_with_visualization()` para llamar a `result.plot()`. Hacia fuera
entrega `DetectionResult` y una imagen PIL; ningún objeto de Ultralytics,
tensor o array escapa como predicción.

## 7. Migración y compatibilidad

### Flujo anterior

```text
Results → YOLODetectionModel → DetectionResponse HTTP
                              → servicios/métricas/dibujo
```

### Flujo resultante

```text
Results → adaptador YOLO → DetectionResult
                         → métricas/dibujo/servicios
                         → presenter HTTP → schemas existentes
```

Cambios de consumidores:

- `BaseObjectDetectionModel` tipa el retorno como `DetectionResult`;
- `YOLODetectionModel` delega toda adaptación;
- inferencia, ataque, evaluación y explicabilidad leen el contrato canónico;
- el dibujo recibe `Sequence[Detection]`;
- los schemas HTTP solo se construyen al preparar cada respuesta.

`to_detection_prediction()` y `to_detection_predictions()` preservan:

- `label` como nombre de clase; si no existe, usan el ID como texto;
- confianza redondeada a cuatro decimales;
- caja pública `x1/y1/x2/y2` redondeada a dos decimales;
- estructura y nombres de campos HTTP anteriores.

El redondeo se ha desplazado a presentación. Las métricas y visualizaciones
trabajan ahora con precisión completa.

Clasificación no se modificó: mantiene su contrato específico por tarea. No se
creó una jerarquía común artificial ni una unión discriminada sin consumidor.

### Correcciones posteriores de compatibilidad

#### Semántica legacy de `attack_success`

La comparación con `git show HEAD:backend/app/ml/models/yolo/model.py` confirmó
la secuencia histórica exacta:

1. `_result_to_response()` ejecutaba `round(confidence, 4)` para cada caja;
2. `calculate_detection_comparison()` recibía esas predicciones ya
   redondeadas;
3. calculaba la media;
4. redondeaba ambas medias a cuatro decimales;
5. evaluaba
   `adversarial_count < original_count or legacy_difference < 0`.

Tras el refactor, el paso 1 desapareció correctamente del modelo, pero la rama
legacy calculaba primero la media de valores completos y solo después la
redondeaba. `mean(round(x, 4))` y `round(mean(x), 4)` pueden clasificar una
muestra de forma diferente.

`_legacy_average_detection_confidence()` restaura únicamente para
`attack_success` la media de confianzas redondeadas individualmente. En
paralelo, `average_original_confidence`,
`average_adversarial_confidence` y `average_confidence_difference` siguen
calculándose con precisión canónica completa. No se convierte a HTTP ni se
modifica el contrato.

#### IDs de clase negativos

`_adapt_row()` valida ahora `class_id >= 0` dentro del bloque que contextualiza
errores de contrato, antes de resolver el nombre. Además,
`_resolve_class_name()` exige defensivamente
`0 <= class_id < len(names)` para catálogos secuenciales.

Un ID `-1` ya no puede acceder al último nombre ni provocar un `IndexError` con
un catálogo vacío. El resultado es siempre un `DetectionAdapterError` como:

```text
native detection 0 is invalid: class_id must not be negative
```

## 8. Escalabilidad y eficiencia

- Tiempo del adaptador: `O(N)` por imagen.
- Memoria adicional: `O(N)` para filas Python y objetos canónicos.
- Transferencias: una conversión tensor/CPU/lista por resultado, no por caja.
- Retención: el contrato no referencia tensor, array ni `Results`.
- Consumidores: no contienen `isinstance`, ramas por modelo ni conocimiento de
  YOLO.
- Nuevo detector: implementaría su adaptador a `DetectionResult` y su clase
  `BaseObjectDetectionModel`; los servicios, métricas, dibujo y API no cambian.
- Batch futuro: cada elemento nativo puede adaptarse independientemente al
  mismo contrato; una operación batch devolvería una colección de
  `DetectionResult` sin alterar la representación por imagen.
- Predicción, metadata de ejecución y explicabilidad siguen separadas. No se
  añadieron campos accidentales al núcleo para soportar visualización.

## 9. Archivos y símbolos

Archivos modificados:

- `backend/app/ml/models/base.py`
  - `BaseObjectDetectionModel.detect`;
  - `BaseObjectDetectionModel.detect_with_visualization`.
- `backend/app/ml/models/yolo/model.py`
  - `YOLODetectionModel._prediction_adapter`;
  - `_result_to_prediction`;
  - retornos canónicos de detección.
- `backend/app/schemas/detection.py`
  - `to_detection_prediction`;
  - `to_detection_predictions`.
- `backend/app/services/detection_service.py`
  - consumo canónico y construcción final de `DetectionResponse`.
- `backend/app/services/detection_attack_service.py`
  - métricas/dibujo canónicos y conversión final.
- `backend/app/services/detection_evaluation_service.py`
  - evaluación canónica y conversión por resultado HTTP.
- `backend/app/services/explainability_service.py`
  - dibujo canónico y conversión final.
- `backend/app/ml/explainability/gradcam.py`
  - `draw_detections_on_image(Sequence[Detection])`.
- `backend/app/utils/detection_metrics.py`
  - `_legacy_average_detection_confidence`;
  - separación explícita entre compatibilidad legacy y métricas canónicas.
- `backend/tests/test_detection_evaluation_service.py`
  - doble migrado a `DetectionResult`.
- `backend/tests/test_detection_metrics.py`
  - regresión del redondeo individual y expectativa legacy actualizada.
- `backend/tests/test_unexpected_service_errors.py`
  - doble migrado a `DetectionResult`.

Archivos nuevos:

- `backend/app/ml/models/detection.py`;
- `backend/app/ml/models/yolo/adapter.py`;
- `backend/tests/test_detection_prediction_contract.py`;
- `backend/tests/test_yolo_detection_adapter.py`;
- `codex-reports/03-refactor-backend-detection-prediction-contract.md`.

## 10. Pruebas añadidas y actualizadas

### Contrato y presentación

Nombres exactos en `test_detection_prediction_contract.py`:

1. `test_detection_result_accepts_empty_prediction_and_is_immutable`
2. `test_detection_result_copies_collection_and_preserves_full_precision`
3. `test_detection_result_accepts_multiple_categories`
4. `test_detection_result_rejects_invalid_source_dimensions`
5. `test_detection_rejects_invalid_confidence`
6. `test_detection_box_rejects_invalid_geometry`
7. `test_detection_result_rejects_boxes_outside_source_bounds`
8. `test_detection_rejects_invalid_class_identity`
9. `test_http_conversion_preserves_legacy_shape_and_rounding`

Los casos parametrizados cubren dimensiones, confianza, `NaN`, infinitos,
coordenadas negativas, cajas invertidas/degeneradas y desbordamiento parcial y
total.

### Adaptador e integración

Nombres exactos en `test_yolo_detection_adapter.py`:

1. `test_ultralytics_adapter_converts_empty_native_result`
2. `test_ultralytics_adapter_converts_actual_results_object`
3. `test_ultralytics_adapter_maps_multiple_detections_and_categories`
4. `test_ultralytics_adapter_preserves_source_coordinates_after_letterboxing`
5. `test_ultralytics_adapter_clips_partially_outside_box`
6. `test_ultralytics_adapter_rejects_invalid_native_detection`
7. `test_ultralytics_adapter_rejects_negative_class_id_without_indexing_names`
8. `test_ultralytics_adapter_rejects_mismatched_source_geometry`
9. `test_ultralytics_adapter_does_not_leak_provider_values`
10. `test_ultralytics_adapter_converts_native_rows_once_per_result`
11. `test_detection_service_uses_yolo_adapter_and_preserves_http_contract`
12. `test_detection_attack_service_keeps_canonical_detections_until_response`
13. `test_detection_explainability_keeps_canonical_detections_until_response`

La prueba 2 instancia un `Results` real de Ultralytics. La 4 usa una fuente
`640 × 320` para demostrar que se preserva la proyección fuente posterior al
letterboxing, sin reescalado duplicado. La 10 verifica exactamente una
transferencia CPU y una conversión a lista para dos cajas. Las pruebas 11–13
ejercitan adaptador/servicios/presentación y consumidores canónicos reales.

La prueba 7 está parametrizada con `names=["person"]` y `names=[]`. Un catálogo
instrumentado registra los accesos y exige que no se invoque `__getitem__`;
ambos casos deben producir `DetectionAdapterError` con el contexto de la fila.

### Compatibilidad de métricas

Prueba añadida:

- `test_legacy_attack_success_rounds_each_confidence_before_averaging`

Prueba actualizada:

- `test_detection_comparison_preserves_legacy_attack_success`

La regresión nueva mantiene dos detecciones en ambas variantes. Las
confianzas completas producen una reducción canónica de `0.000098`, que habría
clasificado el ataque como exitoso con el código defectuoso, mientras que las
dos colecciones históricas redondeadas son idénticas:
`[0.5000, 0.5001]`. Por ello exige `attack_success is False` y conserva el
delta canónico completo.

También se actualizó
`test_detection_evaluation_uses_canonical_deltas_and_legacy_success` para que
el doble produzca el contrato verdadero. Los seis tests de
`test_unexpected_service_errors.py` conservaron sus aserciones y ahora usan un
doble de detección canónico.

## 11. Comandos y resultados reales

### Línea base

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detect.py tests/test_detection_attack.py \
  tests/test_detection_evaluation.py \
  tests/test_detection_evaluation_service.py \
  tests/test_detection_metrics.py tests/test_explainability.py \
  tests/test_models.py tests/test_yolo_confidence_attack.py \
  tests/test_unexpected_service_errors.py
```

Resultado: `81 passed in 1.23s`.

### Pruebas focalizadas

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_prediction_contract.py \
  tests/test_yolo_detection_adapter.py
```

Resultado final: `39 passed in 0.58s`.

La ejecución focalizada ampliada durante la migración terminó con
`116 passed in 2.40s`; después se añadieron cuatro casos de regresión y la
suite final completa los cubrió.

Línea base específica de esta corrección:

```text
docker compose exec backend python -m pytest -ra -W always \
  tests/test_detection_metrics.py tests/test_yolo_detection_adapter.py \
  tests/test_detection_prediction_contract.py tests/test_detection_attack.py \
  tests/test_detection_evaluation_service.py
```

Resultado antes de editar: `66 passed in 0.99s`.

Resultado después de corregir: `69 passed in 1.30s`.

### Suite completa final

```text
docker compose exec backend python -m pytest -ra -W always
```

Resultado final:

```text
449 collected
447 passed, 2 skipped in 27.56s
```

Omisiones:

- CUDA no disponible en
  `test_attack_source_geometry_reconstruction.py:148`;
- CUDA no disponible en `test_gradcam_source_geometry.py:240`.

### Ruff, formato y compilación

```text
docker compose exec backend python -m ruff check app tests
docker compose exec backend python -m ruff format --check app tests
docker compose exec backend python -m compileall -q app tests
git diff --check
```

Resultados finales:

- Ruff: `All checks passed!`;
- formato: `114 files already formatted`;
- `compileall`: código 0, sin salida;
- `git diff --check`: código 0, sin salida.

En esta corrección, el primer `ruff format --check` identificó
`app/utils/detection_metrics.py` y `tests/test_detection_metrics.py`. Se
ejecutó `ruff format` sobre ambos; todas las comprobaciones posteriores
pasaron.

### Incidencias de ejecución

Dos invocaciones iniciales de Docker Compose se lanzaron desde la raíz del
repositorio, donde no existe el YAML, y devolvieron
`no configuration file provided`. Se repitieron desde `backend/` y pasaron.

Dos intentos auxiliares que no coincidían con el prefijo de ejecución aprobado
(`python -c` y un `pytest --collect-only` sin las opciones habituales) fueron
bloqueados por `snap-confine`. No eran validaciones necesarias: el objeto real
se cubrió mediante pytest y los nombres se verificaron en los archivos.

## 12. Validaciones no ejecutadas

- No existe configuración ni dependencia de mypy, pyright u otro type checker
  en `pyproject.toml`, `requirements.txt` o `pytest.ini`; se ejecutó
  `compileall` como comprobación aplicable.
- No se ejecutó inferencia descargando pesos reales `yolov8n.pt`, porque
  implicaría un artefacto externo y no es necesaria para validar la frontera.
  Sí se utilizó la clase `Results` real, tensores reales y el modelo real con
  `_run_inference` sustituido en la integración.
- No se ejecutaron tests frontend: no hubo cambios frontend y están fuera del
  alcance solicitado.
- Los dos tests CUDA fueron recopilados, pero pytest los omitió al no haber GPU.

## 13. Diff final completo y alcance

Con la corrección posterior incluida, `git diff --stat` para archivos
rastreados es:

```text
12 files changed, 197 insertions(+), 89 deletions(-)
```

Los cuatro archivos de código/tests nuevos suman 910 líneas. El informe es el
quinto archivo nuevo.

Estado completo:

```text
M  backend/app/ml/explainability/gradcam.py
M  backend/app/ml/models/base.py
M  backend/app/ml/models/yolo/model.py
M  backend/app/schemas/detection.py
M  backend/app/services/detection_attack_service.py
M  backend/app/services/detection_evaluation_service.py
M  backend/app/services/detection_service.py
M  backend/app/services/explainability_service.py
M  backend/app/utils/detection_metrics.py
M  backend/tests/test_detection_evaluation_service.py
M  backend/tests/test_detection_metrics.py
M  backend/tests/test_unexpected_service_errors.py
?? backend/app/ml/models/detection.py
?? backend/app/ml/models/yolo/adapter.py
?? backend/tests/test_detection_prediction_contract.py
?? backend/tests/test_yolo_detection_adapter.py
?? codex-reports/03-refactor-backend-detection-prediction-contract.md
```

La revisión manual del diff confirmó:

- no quedan imports de schemas HTTP de detección en la abstracción ni en la
  implementación de detección bajo `ml/models`;
- no queda el camino obsoleto `_result_to_response`;
- servicios y dibujo no acceden a `bbox`/`label` legacy;
- la conversión HTTP aparece solo al construir respuestas;
- la precisión completa sigue alimentando promedios y deltas canónicos;
- solo `attack_success` usa el redondeo individual legacy;
- un ID negativo se rechaza antes de resolver cualquier nombre;
- no hay cambios de dependencias, API pública, frontend ni clasificación.

## 14. Riesgos y pendientes

- La geometría fuente depende de la garantía de postproceso de
  `Ultralytics Results`; la comprobación estricta de `orig_shape` hace visible
  cualquier cambio incompatible del proveedor.
- Un detector que entregue coordenadas en su tensor de entrada necesitará
  conservar en su adaptador la transformación completa de preprocesado y
  deshacer resize/padding antes de construir `DetectionResult`.
- El clipping parcial preserva la tolerancia práctica a pequeñas salidas fuera
  del borde. Si en el futuro se desea medir la incidencia, debe hacerse como
  telemetría del adaptador, no añadiendo campos específicos al contrato.
- Los errores del adaptador permanecen internos y siguen el manejo genérico
  actual. No se introdujo un nuevo error HTTP.
- La compatibilidad legacy de `attack_success` es deliberadamente distinta del
  signo del delta canónico; debe mantenerse aislada mientras ese booleano forme
  parte del contrato observable.
- Batch, métricas IoU/mAP y asociación con anotaciones quedan fuera de alcance.

## 15. Checklist de aceptación

- Cumplido — contrato interno canónico, tipado e inmutable.
- Cumplido — dimensiones fuente, resultado vacío, clase, nombre, confianza y
  caja `xyxy`.
- Cumplido — invariantes y errores internos tipados.
- Cumplido — política de clipping/rechazo centralizada y probada.
- Cumplido — adaptador real de Ultralytics en un único punto.
- Cumplido — una conversión tensor/CPU/lista por resultado.
- Cumplido — no escapan tensores, arrays, `Results` ni formatos posicionales.
- Cumplido — modelo y consumidores usan el contrato como frontera real.
- Cumplido — métricas y dibujo reciben precisión completa.
- Cumplido — `attack_success` reproduce el redondeo individual histórico sin
  reducir la precisión de promedios ni deltas canónicos.
- Cumplido — IDs negativos no indexan catálogos vacíos ni no vacíos y producen
  `DetectionAdapterError` contextualizado.
- Cumplido — schemas HTTP y semántica observable preservados.
- Cumplido — clasificación y Grad-CAM existentes conservados.
- Cumplido — dobles legacy migrados sin reducir regresiones.
- Cumplido — pruebas obligatorias de contrato, adaptador e integración.
- Cumplido — suite backend, Ruff, formato y compilación correctos.
- Cumplido — sin dependencias, migraciones, endpoints ni frontend.
- Cumplido — diff revisado, incluidos archivos no rastreados.
- Cumplido — no se hizo commit ni push.
