# Informe: semántica base de evaluación de detección

## 1. Estado

- Resultado: completado.
- Resumen breve: se unificaron los deltas básicos de detección como
  `original - adversarial` en el cálculo por imagen y agregado, se mantuvo sin
  cambios la clasificación legacy de éxito, se añadió una versión aditiva del
  contrato y se aclararon los textos del frontend.
- Commit base o hash de `HEAD`:
  `dc97b5286447771b6a4111192be343a55a7ab84b`.
- El árbol inicial estaba limpio. No contenía cambios previos del usuario.

## 2. Diagnóstico inicial

El flujo real encontrado fue:

1. `backend/app/main.py` registra las rutas de detección.
2. `backend/app/api/routes/detection_attack.py` expone `POST /detect/attack` y
   delega en `apply_detection_attack`.
3. `backend/app/api/routes/detection_evaluate.py` expone
   `POST /detect/evaluate` y delega en `evaluate_detection_dataset`.
4. `backend/app/schemas/detection_attack.py` y
   `backend/app/schemas/detection_evaluation.py` definen los contratos
   Pydantic que serializa FastAPI.
5. `backend/app/services/detection_attack_service.py` y
   `backend/app/services/detection_evaluation_service.py` obtienen predicciones
   originales/adversarias y consumen la utilidad compartida.
6. `backend/app/utils/detection_metrics.py`, en
   `calculate_detection_comparison`, calcula los conteos, confianzas medias,
   deltas y éxito legacy por imagen.
7. `evaluate_detection_dataset` vuelve a calcular el delta agregado de
   conteos y promedia los deltas de confianza por imagen.
8. `frontend/src/api/detect.js` y
   `frontend/src/api/detectionEvaluation.js` recuperan las respuestas sin
   transformar sus métricas.
9. `frontend/src/pages/DetectionView.jsx` y
   `frontend/src/components/DetectionEvaluationPanel.jsx` presentan
   directamente los campos del backend.

No existe un `AGENTS.md` aplicable al proyecto. El único encontrado está
dentro de `frontend/node_modules/recharts`, una dependencia que no se modificó.

Ubicaciones y convenciones iniciales:

- `backend/app/utils/detection_metrics.py`,
  `calculate_detection_comparison`:
  `detections_difference = adversarial_count - original_count` y
  `average_confidence_difference = adversarial_average - original_average`.
- `backend/app/services/detection_evaluation_service.py`,
  `evaluate_detection_dataset`: el resumen repetía
  `total_adversarial_detections - total_original_detections`.
- `backend/tests/test_detection_metrics.py`: las aserciones unitarias
  confirmaban expresamente esa convención `adversarial - original`.
- `backend/tests/test_detection_evaluation.py` y
  `backend/tests/test_detection_attack.py`: los fixtures HTTP ya representaban
  las caídas de confianza y detecciones con valores positivos, es decir,
  `original - adversarial`.
- Los tests HTTP sustituían los servicios completos por mocks, por lo que no
  detectaban la contradicción entre fixture y cálculo real.
- El frontend no invertía el signo: mostraba directamente
  `detections_difference` y `average_confidence_difference`. Sus etiquetas
  eran ambiguas y llamaban “éxito” a la métrica sin identificarla como legacy.

La contradicción era, por tanto, real: utilidad y agregado usaban un signo,
mientras los fixtures/respuestas simuladas ya comunicaban el contrario.

## 3. Semántica final

Las fórmulas canónicas son:

```text
average_confidence_difference =
    original_average_confidence - adversarial_average_confidence

detections_difference =
    original_detection_count - adversarial_detection_count
```

La API conserva el nombre público existente
`average_confidence_difference`; este corresponde al
`confidence_difference` descrito en el encargo.

| Campo | Positivo | Cero | Negativo |
|---|---|---|---|
| `average_confidence_difference` | Baja la confianza adversaria | No cambia | Sube la confianza adversaria |
| `detections_difference` | Hay menos detecciones adversarias | Misma cantidad | Hay más detecciones adversarias |

`detections_difference` solo describe un cambio de cantidad. Sin ground truth
no determina si las detecciones eliminadas o añadidas eran correctas y no se
interpreta como mejora o degradación de calidad.

La definición exacta de éxito legacy se conserva:

```text
adversarial_count < original_count
OR
round(
    round(adversarial_average_confidence, 4)
    - round(original_average_confidence, 4),
    4
) < 0
```

La tasa legacy agregada sigue siendo:

```text
successful_images_under_legacy_rule / total_images
```

con `0.0` cuando el dataset no produce resultados. Esta regla permanece
separada del signo canónico nuevo, de modo que invertir los deltas no
reclasifica muestras.

El contrato se versiona de forma aditiva en el nivel superior de
`DetectionAttackResponse` y `DetectionEvaluationResponse`, reutilizando la
convención de las métricas avanzadas de clasificación:

- `metrics_schema_version`: `str`, valor actual `"1.0"`.
- `delta_convention`:
  `Literal["original_minus_adversarial"]`, valor
  `"original_minus_adversarial"`.

Ejemplo representativo de `POST /detect/evaluate`:

```json
{
  "metrics_schema_version": "1.0",
  "delta_convention": "original_minus_adversarial",
  "model_id": "detector-1",
  "model_name": "Detector",
  "dataset_id": "dataset-1",
  "attack_type": "random_noise",
  "parameters": {
    "epsilon": 0.03,
    "alpha": 0.01,
    "steps": 10
  },
  "summary": {
    "total_images": 1,
    "total_original_detections": 2,
    "total_adversarial_detections": 1,
    "detections_difference": 1,
    "average_original_detections_per_image": 2.0,
    "average_adversarial_detections_per_image": 1.0,
    "average_original_confidence": 0.85,
    "average_adversarial_confidence": 0.5,
    "average_confidence_difference": 0.35,
    "images_with_reduced_detections": 1,
    "images_with_no_adversarial_detections": 0,
    "attack_success_rate": 1.0,
    "average_perturbation_linf": 0.03,
    "average_perturbation_l2": 0.5
  },
  "results": [
    {
      "image_id": "image-1",
      "filename": "image-1.png",
      "original_detection_count": 2,
      "adversarial_detection_count": 1,
      "detections_difference": 1,
      "average_original_confidence": 0.85,
      "average_adversarial_confidence": 0.5,
      "average_confidence_difference": 0.35,
      "attack_success": true,
      "perturbation_linf": 0.03,
      "perturbation_l2": 0.5,
      "original_detections": [
        {
          "label": "car",
          "confidence": 0.9,
          "bbox": {"x1": 10, "y1": 20, "x2": 120, "y2": 160}
        },
        {
          "label": "person",
          "confidence": 0.8,
          "bbox": {"x1": 40, "y1": 30, "x2": 90, "y2": 170}
        }
      ],
      "adversarial_detections": [
        {
          "label": "car",
          "confidence": 0.5,
          "bbox": {"x1": 10, "y1": 20, "x2": 120, "y2": 160}
        }
      ]
    }
  ]
}
```

## 4. Cambios realizados

### Backend

- `backend/app/schemas/detection_metrics.py`
  - Símbolo: `DetectionMetricsContract`.
  - Motivo: centralizar la versión, convención y documentación semántica.
  - Antes: detección no tenía versión explícita ni convención declarada.
  - Ahora: aporta `metrics_schema_version = "1.0"` y
    `delta_convention = "original_minus_adversarial"` y documenta todos los
    signos, la cautela sobre conteos y el principio para métricas futuras.

- `backend/app/schemas/detection_attack.py`
  - Símbolos: `DetectionAttackComparison`, `DetectionAttackResponse`.
  - Motivo: documentar el carácter legacy del éxito y versionar la respuesta.
  - Antes: respuesta sin metadatos de contrato y comparación sin semántica
    explícita.
  - Ahora: respuesta hereda el contrato común y el schema documenta deltas y
    éxito legacy.

- `backend/app/schemas/detection_evaluation.py`
  - Símbolos: `DetectionEvaluationImageResult`,
    `DetectionEvaluationSummary`, `DetectionEvaluationResponse`.
  - Motivo: documentar deltas/ASR legacy y versionar la respuesta.
  - Antes: campos sin explicación formal ni versión.
  - Ahora: docstrings cercanos al contrato y metadatos aditivos comunes.

- `backend/app/utils/detection_metrics.py`
  - Símbolos: `average_detection_confidence`,
    `calculate_detection_comparison`.
  - Motivo: establecer el único cálculo canónico y evitar redondeo previo.
  - Antes: promedios redondeados antes del delta y ambos deltas calculados
    como `adversarial - original`.
  - Ahora: medias y delta se calculan con precisión interna, ambos deltas usan
    `original - adversarial` y la regla legacy se calcula por separado con el
    redondeo histórico.

- `backend/app/services/detection_attack_service.py`
  - Símbolo: `apply_detection_attack`.
  - Motivo: situar el redondeo en el límite de presentación.
  - Antes: recibía valores ya redondeados desde la utilidad.
  - Ahora: redondea a cuatro decimales al construir el DTO público.

- `backend/app/services/detection_evaluation_service.py`
  - Símbolo: `evaluate_detection_dataset`.
  - Motivo: corregir el agregado y redondear solo al construir resultados.
  - Antes: el resumen de conteos repetía `adversarial - original`.
  - Ahora: usa `original - adversarial`, agrega deltas internos sin redondeo
    previo y redondea la representación pública.

### Frontend

- `frontend/src/components/DetectionEvaluationPanel.jsx`
  - Símbolo: `DetectionEvaluationPanel`.
  - Motivo: hacer explícitos fórmula, signo y condición legacy.
  - Antes: etiquetas “Diferencia” y “Tasa de éxito del ataque” sin dirección.
  - Ahora: etiquetas `original − adversaria`, explicación de los signos,
    cautela sobre falta de ground truth y denominación “legacy”. Los valores se
    siguen leyendo directamente del backend.

- `frontend/src/pages/DetectionView.jsx`
  - Símbolo: `DetectionView`.
  - Motivo: alinear la vista de ataque individual con el mismo contrato.
  - Antes: etiquetas ambiguas.
  - Ahora: fórmula visible, resultado identificado como legacy y explicación
    que evita interpretar el conteo como calidad.

### Tests

- `backend/tests/test_detection_metrics.py`
  - Símbolos: tests de `calculate_detection_comparison`.
  - Motivo: regresión completa de signos, vacíos, valores finitos, precisión y
    preservación legacy.
  - Antes: seis tests, varios afirmando el signo contrario.
  - Ahora: veinte casos recogidos por pytest, incluidos ocho escenarios
    parametrizados de equivalencia con la regla legacy.

- `backend/tests/test_detection_evaluation_service.py`
  - Símbolo:
    `test_detection_evaluation_uses_canonical_deltas_and_legacy_success`.
  - Motivo: probar el servicio real y su agregado, no solo una ruta mockeada.
  - Antes: no había prueba feliz directa del servicio.
  - Ahora: verifica deltas positivos/negativos, resumen, tasa legacy y versión.

- `backend/tests/test_detection_evaluation.py`
  - Símbolos: fixtures y aserciones de las respuestas HTTP de evaluación.
  - Motivo: alinear el contrato HTTP con la semántica y la versión.
  - Antes: deltas canónicos en fixtures, pero sin versión ni aserciones
    explícitas de los deltas principales.
  - Ahora: fixtures versionados y aserciones de fórmula/valor.

- `backend/tests/test_detection_attack.py`
  - Símbolos: fixtures y aserciones de las respuestas HTTP de ataque.
  - Motivo: versionar y comprobar el segundo consumidor del cálculo común.
  - Antes: sin metadatos de contrato.
  - Ahora: fixtures y respuesta verifican versión y convención.

### Documentación

- Este informe.
- La documentación cercana al contrato está en
  `backend/app/schemas/detection_metrics.py`; la documentación cercana al
  cálculo está en `calculate_detection_comparison`.

## 5. Compatibilidad

- Se preservan todos los campos públicos existentes:
  `detections_difference`, `average_confidence_difference`,
  `attack_success`, `attack_success_rate` y el resto de los DTO.
- No se renombró ni eliminó ningún campo.
- Los únicos campos estructurales nuevos son aditivos:
  `metrics_schema_version` y `delta_convention`.
- La semántica de los dos deltas cambia intencionadamente para corregir el
  contrato. Un cliente que dependiera del signo real anterior deberá leer la
  versión/convención nueva; los fixtures HTTP y la intención de interfaz ya
  estaban orientados a la convención final.
- La tasa legacy conserva su comportamiento mediante una condición separada
  que reproduce el conteo y el redondeo anteriores. Se comprobó con
  `test_detection_comparison_preserves_legacy_attack_success` sobre ocho
  combinaciones y con el test de servicio.
- No hay persistencia afectada ni se requiere migración de datos.

## 6. Decisiones tomadas

### Metadatos en el nivel superior de ambas respuestas

- Decisión: ubicar `metrics_schema_version` y `delta_convention` en
  `DetectionAttackResponse` y `DetectionEvaluationResponse`.
- Alternativas: repetirlos en cada resultado/resumen o crear un nuevo objeto
  `advanced_metrics`.
- Razón: detección todavía no tiene el contenedor avanzado de clasificación;
  el nivel superior es un único límite de contrato sin duplicar metadatos.
- Impacto futuro: una futura estructura avanzada podrá mantener estos campos o
  versionar un objeto anidado sin alterar los campos legacy actuales.

### Reutilizar nombres de clasificación

- Decisión: usar exactamente `metrics_schema_version` y
  `delta_convention = "original_minus_adversarial"`.
- Alternativas: `contract_version`, `schema_version` u otro sistema.
- Razón: las métricas avanzadas de clasificación ya establecen esa convención.
- Impacto futuro: un solo vocabulario de versionado para clientes y tests.

### Precisión interna y redondeo de API

- Decisión: calcular medias/deltas sin redondear y redondear a cuatro
  decimales al construir el DTO público.
- Alternativas: conservar el redondeo dentro de la utilidad o dejar todos los
  decimales en la API.
- Razón: evita redondear antes del cálculo y conserva la presentación histórica
  de cuatro decimales.
- Impacto futuro: cálculos agregados pueden reutilizar valores precisos.

### Regla legacy desacoplada del delta canónico

- Decisión: reproducir expresamente la comparación histórica redondeada.
- Alternativas: definir éxito como `average_confidence_difference > 0` o usar
  medias sin redondear.
- Razón: ambas alternativas podrían reclasificar casos limítrofes y violar la
  compatibilidad exigida.
- Impacto futuro: la regla queda visible y podrá retirarse de forma consciente
  cuando se implemente análisis adversario por objetos.

### Incluir ataque individual y evaluación de dataset

- Decisión: aplicar contrato/documentación a `/detect/attack` y
  `/detect/evaluate`.
- Alternativa: tocar solo la evaluación de dataset.
- Razón: ambos exponen los mismos campos y consumen la misma función; dejar uno
  fuera mantendría dos contratos incompatibles.
- Impacto futuro: una sola semántica en todos los consumidores actuales.

## 7. Pruebas y validación

| Momento | Comando exacto | Propósito | Resultado |
|---|---|---|---|
| Previo | `../venv/bin/python -m pytest -q tests/test_detection_metrics.py tests/test_detection_evaluation.py tests/test_detection_attack.py` | Línea base local | Interrumpido: el fixture `autouse` de `tests/conftest.py` intenta limpiar MongoDB en `mongo:27017`, no accesible fuera de Compose. No se usó como resultado de tests. |
| Previo | `docker compose exec backend python -m pytest -ra -W always tests/test_detection_metrics.py tests/test_detection_evaluation.py tests/test_detection_attack.py` | Línea base en el entorno previsto | 18 superados, 0 fallidos. |
| Intermedio | `docker compose exec backend python -m pytest -ra -W always tests/test_detection_metrics.py tests/test_detection_evaluation_service.py tests/test_detection_evaluation.py tests/test_detection_attack.py` | Primera ejecución con tests nuevos | 31 superados, 2 fallidos en aserciones nuevas; se corrigieron una comparación exacta de coma flotante y unos datos que activaban legítimamente la regla legacy. |
| Final focalizado | `docker compose exec backend python -m pytest -ra -W always tests/test_detection_metrics.py tests/test_detection_evaluation_service.py tests/test_detection_evaluation.py tests/test_detection_attack.py` | Cálculo, servicio, HTTP y contrato de detección | 33 superados, 0 fallidos. Repetido tras el último ajuste con el mismo resultado final. |
| Final backend | `docker compose exec backend python -m pytest -ra -W always` | Suite completa de regresión | 364 superados, 0 fallidos, 2 omitidos por falta de CUDA; 366 recogidos. |
| Final lint Python | `docker compose exec backend python -m ruff check app/schemas/detection_metrics.py app/schemas/detection_attack.py app/schemas/detection_evaluation.py app/utils/detection_metrics.py app/services/detection_attack_service.py app/services/detection_evaluation_service.py tests/test_detection_metrics.py tests/test_detection_evaluation_service.py tests/test_detection_evaluation.py tests/test_detection_attack.py` | Lint de Python modificado | Correcto: `All checks passed!`. |
| Final formato Python | `docker compose exec backend python -m ruff format --check app/schemas/detection_metrics.py app/schemas/detection_attack.py app/schemas/detection_evaluation.py app/utils/detection_metrics.py app/services/detection_attack_service.py app/services/detection_evaluation_service.py tests/test_detection_metrics.py tests/test_detection_evaluation_service.py tests/test_detection_evaluation.py tests/test_detection_attack.py` | Formato de Python modificado | Correcto: 10 archivos ya formateados. |
| Final test frontend | `npm run test:gradcam` | Suite automatizada disponible del frontend | 1 test/suite superado, 0 fallidos. |
| Final build frontend | `npm run build` | Compilación de producción | Correcto; 631 módulos transformados. Advertencia no bloqueante por chunk de 732.37 kB. |
| Final lint frontend | `npm run lint` | Lint global del frontend | No pasa: 11 errores y 1 advertencia preexistentes. Los dos errores en archivos tocados están en `DetectionEvaluationPanel.jsx:65` y `DetectionView.jsx:69`, líneas de efectos idénticas en `HEAD`; los restantes están en archivos no modificados. |
| Final diff | `git diff --check` | Espacios y conflictos de parche | Correcto, sin salida. |

## 8. Casos cubiertos

1. Confianza adversaria disminuye:
   `test_detection_comparison_detects_confidence_reduction`.
2. Confianza adversaria aumenta:
   `test_detection_comparison_reports_unsuccessful_attack`.
3. Confianza no cambia:
   `test_detection_comparison_reports_unchanged_confidence_and_count`.
4. Desaparecen detecciones:
   `test_detection_comparison_detects_reduced_count`.
5. Aparecen detecciones:
   `test_detection_comparison_reports_new_detections_with_negative_delta`.
6. Cantidad sin cambios:
   `test_detection_comparison_reports_unchanged_confidence_and_count` y los
   tests de reducción/aumento de confianza.
7. Ambas variantes sin detecciones:
   `test_detection_comparison_handles_no_detections`, incluida comprobación
   `math.isfinite`.
8. Solo una variante con detecciones:
   `test_detection_comparison_handles_detections_on_only_one_variant`,
   parametrizado en ambas direcciones y con comprobación finita.
9. Respuesta HTTP, cálculo y fixtures alineados:
   `test_detection_evaluation_uses_canonical_deltas_and_legacy_success`,
   `test_detect_evaluate_returns_detection_evaluation_response` y
   `test_detect_attack_returns_detection_attack_response`.
10. Clasificación legacy preservada:
    `test_detection_comparison_preserves_legacy_attack_success`, parametrizado
    sobre ocho entradas, y el test del servicio.
11. Versión con tipo/valor esperados:
    tests de servicio y de ambos endpoints comprueban la cadena `"1.0"` y
    `"original_minus_adversarial"`.
12. Frontend sin reinversión:
    `DetectionEvaluationPanel` y `DetectionView` renderizan directamente los
    campos del backend; las etiquetas declaran `original − adversaria`.
    `npm run build` valida ambos consumidores.

## 9. Diff y alcance

`git diff --stat` para archivos ya seguidos por Git:

```text
10 files changed, 253 insertions(+), 32 deletions(-)
```

Hay además tres archivos nuevos no incluidos por `git diff --stat` mientras no
estén en el índice:

- `backend/app/schemas/detection_metrics.py` (28 líneas).
- `backend/tests/test_detection_evaluation_service.py` (110 líneas).
- `codex-reports/01-fix-detection-evaluation-semantics.md` (este informe).

Lista final de archivos modificados o nuevos:

- `backend/app/schemas/detection_attack.py`
- `backend/app/schemas/detection_evaluation.py`
- `backend/app/schemas/detection_metrics.py`
- `backend/app/services/detection_attack_service.py`
- `backend/app/services/detection_evaluation_service.py`
- `backend/app/utils/detection_metrics.py`
- `backend/tests/test_detection_attack.py`
- `backend/tests/test_detection_evaluation.py`
- `backend/tests/test_detection_evaluation_service.py`
- `backend/tests/test_detection_metrics.py`
- `frontend/src/components/DetectionEvaluationPanel.jsx`
- `frontend/src/pages/DetectionView.jsx`
- `codex-reports/01-fix-detection-evaluation-semantics.md`

No se hizo commit ni push. No había cambios previos del usuario que preservar.
El build no modificó artefactos seguidos por Git. No hubo cambios fuera de
alcance. Los servicios Docker usados para las pruebas se detuvieron al terminar,
restaurando su estado inicial.

## 10. Riesgos y asuntos pendientes

- Riesgo conocido: aunque el contrato es estructuralmente compatible, la
  corrección del signo cambia intencionadamente el significado de valores que
  provinieran del cálculo real anterior. La versión y `delta_convention`
  permiten detectarlo.
- La tasa de éxito continúa siendo una heurística legacy sin ground truth,
  exactamente como exige esta sub-issue. Su sustitución por análisis por
  objetos queda pendiente para una tarea posterior.
- `npm run lint` está bloqueado por deuda preexistente: 11 errores y 1
  advertencia. No se corrigieron porque implicaría refactorizar efectos y
  contextos fuera del alcance. El build y el test frontend sí pasan.
- Dos tests GPU de la suite backend se omiten porque CUDA no está disponible:
  `test_attack_source_geometry_reconstruction.py:148` y
  `test_gradcam_source_geometry.py:240`.
- No se encontraron otros bugs independientes en el flujo inspeccionado.
- Para futuras métricas de error se deberá declarar dirección o usar nombres
  inequívocos, según queda documentado en el contrato.

## 11. Checklist de aceptación

- Cumplido — Existe una única convención de signo en todo el flujo.
  Evidencia: cálculo compartido y agregado usan `original - adversarial`; la
  búsqueda final no encontró las fórmulas contrarias.
- Cumplido — Backend, API, fixtures, tests y frontend están alineados.
  Evidencia: test del servicio, tests HTTP, fixtures versionados y consumidores
  directos; suite focalizada con 33 tests.
- Cumplido — Los signos positivo, cero y negativo tienen significado
  documentado. Evidencia: `DetectionMetricsContract`, docstring del cálculo y
  etiquetas/explicaciones del frontend.
- Cumplido — `detections_difference` no se presenta como calidad real sin
  ground truth. Evidencia: schema y ambas vistas lo declaran expresamente.
- Cumplido — La métrica de éxito actual está marcada como legacy y no ha sido
  redefinida. Evidencia: schemas, frontend y test parametrizado de equivalencia.
- Cumplido — El contrato dispone de una versión explícita y compatible.
  Evidencia: `metrics_schema_version = "1.0"` y
  `delta_convention = "original_minus_adversarial"` en ambas respuestas.
- Cumplido — No se han eliminado ni renombrado campos existentes. Evidencia:
  cambio exclusivamente aditivo en schemas.
- Cumplido — Los casos límite no producen errores ni valores no finitos.
  Evidencia: tests de ambas variantes vacías y de una sola variante con
  `math.isfinite`.
- Cumplido — Los tests relevantes pasan o sus bloqueos quedan demostrados.
  Evidencia: 33 tests focalizados y 364 tests de backend pasan; build y test
  frontend pasan; lint global documentado con líneas preexistentes.
- Cumplido — El diff se limita al alcance de esta sub-issue. Evidencia: 13
  archivos listados, todos de cálculo, contrato, consumidores, tests o informe.
- Cumplido — Se ha generado el informe de supervisión solicitado. Evidencia:
  `codex-reports/01-fix-detection-evaluation-semantics.md`.
