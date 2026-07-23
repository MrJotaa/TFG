# Grad-CAM Subissue 1 Report

## Alcance implementado

Se separaron explícitamente las dos representaciones del Grad-CAM limpio solicitado
desde Predict:

- `model_input`: tensor devuelto por el `preprocess` vigente del modelo y usado en el
  único `forward`, el `backward`, los hooks, las activaciones y los gradientes.
- `source_image`: copia RGB de la imagen recuperada del dataset, sin redimensionado,
  crop, normalización ni reconstrucción desde un tensor, usada únicamente para la
  representación.

El mapa normalizado se proyecta directamente a `source_image.size` y la imagen final se
construye en modo RGB. La nueva entrada explícita se integra únicamente en
`/explain/gradcam`. El contrato legacy `generate_gradcam(model_wrapper, image)` que
consumen los ataques se conserva.

También se garantizó mediante `finally` que los hooks se retiren incluso cuando falle el
forward, que las listas locales de activaciones y gradientes se vacíen y que los
gradientes de parámetros del modelo queden a `None`.

No se modificaron Grad-CAM adversarial, Evaluate, frontend, ataques ni detección.

## Flujo anterior

`POST /explain/gradcam` recibía `dataset_id` e `image_id`, recuperaba una imagen RGB
desde el archivo original del dataset y llamaba a `generate_gradcam(model, image)`.

Ese helper usaba un único argumento `image` para dos responsabilidades:

1. Ejecutaba `model_wrapper.preprocess(image)` y alimentaba el tensor resultante al
   modelo.
2. Redimensionaba el CAM a `image.size` y mezclaba el heatmap con
   `image.convert("RGB")`.

En el estado inicial inspeccionado, el endpoint ya recuperaba el archivo fuente sin
resize y el helper redimensionaba el CAM a ese PIL. Por ello, la premisa de una salida
224×224 no era reproducible directamente en este checkout para `/explain/gradcam`.
El problema arquitectónico comprobable era el acoplamiento: la API del helper no
diferenciaba ni garantizaba qué representación era científica y cuál era visual.

Los hooks se retiraban solamente al final del camino exitoso, no dentro de un bloque
`finally`.

## Flujo resultante

El flujo limpio queda así:

1. `get_image_from_dataset(dataset_id, image_id)` recupera la imagen solicitada.
2. El servicio crea `source_image = image.copy().convert("RGB")`.
3. El servicio crea una sola vez `model_input = model.preprocess(image)`.
4. `generate_clean_gradcam(model, model_input, source_image)`:
   - registra los mismos hooks sobre la misma capa objetivo;
   - realiza un único forward con `model_input`;
   - elige la clase predicha con el mismo `argmax`;
   - realiza backward sobre el logit de esa clase;
   - calcula y normaliza el CAM con la misma fórmula;
   - proyecta el mapa a `(source_width, source_height)`;
   - aplica el mismo colormap y mezcla sobre la copia RGB fuente.
5. El servicio persiste `gradcam.png`, conserva los mismos metadatos y devuelve el
   mismo `GradCamResponse`.
6. En todos los caminos se retiran hooks y se liberan referencias locales a
   activaciones y gradientes.

No se realiza una inferencia adicional para adaptar la geometría.

## Archivos modificados

- `backend/app/ml/explainability/gradcam.py`
  - Se separaron cálculo y renderizado en helpers internos.
  - Se añadió `generate_clean_gradcam`, cuya API exige el tensor preprocesado y la
    imagen fuente por separado.
  - Se conservó `generate_gradcam` como wrapper legacy para Attack.
  - Se añadió limpieza segura de hooks, gradientes y referencias locales.
- `backend/app/services/explainability_service.py`
  - `/explain/gradcam` ahora prepara explícitamente `model_input` y `source_image`, y
    llama a `generate_clean_gradcam`.
  - No se alteraron persistencia, metadatos ni respuesta.
- `backend/tests/test_gradcam_source_geometry.py`
  - Se añadieron pruebas focalizadas de geometría, PNG persistido, entrada científica,
    fondo visual, semántica de mezcla, número de forwards, clase objetivo, recursos,
    contrato de ruta y compatibilidad legacy.
- `backend/tests/test_unexpected_service_errors.py`
  - Se adaptó el doble del servicio a la nueva función limpia y se dotó al modelo fake
    del `preprocess` que ahora ejecuta explícitamente el servicio.
- `GRADCAM_SUBISSUE_1_REPORT.md`
  - Informe obligatorio de esta subissue.

## Contrato entre entrada del modelo e imagen fuente

`model_input` procede exclusivamente de `model.preprocess(image)`. No se obtiene de la
imagen de salida, de un PNG anterior ni de una reconstrucción. Forward y backward
reciben ese mismo objeto tensor; las pruebas verifican identidad, geometría distinta de
la fuente y una única ejecución.

`source_image` es `image.copy().convert("RGB")`. No pasa por el preprocess y no se
redimensiona. El renderer vuelve a copiarla antes de convertirla a un array, por lo que
ni tamaño, modo ni bytes de la imagen entregada se mutan.

La imagen fuente solo se consulta después de obtener el CAM normalizado. Nunca se
alimenta directamente al modelo sin preprocess.

## Semántica visual conservada

La secuencia exacta es:

1. Pesos por canal:
   `weights = gradient.mean(dim=(2, 3), keepdim=True)`.
2. Combinación:
   `cam = (weights * activation).sum(dim=1).squeeze()`.
3. Rectificación:
   `cam = relu(cam)`.
4. Normalización:
   `cam = cam - min(cam)` y después
   `cam = cam / (max(cam) + 1e-8)`.
5. Cuantización previa al resize:
   `uint8(cam * 255)`, con truncado de NumPy.
6. Proyección:
   `PIL.Image.resize(source_image.size)`. Se conserva la llamada sin argumento
   `resample`; para la imagen `L` empleada, Pillow 12.2.0 usa su filtro por defecto
   `BICUBIC`.
7. Colormap:
   `matplotlib.cm.jet(resized_cam / 255.0)[:, :, :3]`.
8. Cuantización del heatmap:
   `uint8(heatmap * 255)`.
9. Orden de canales:
   RGB tanto en el heatmap como en la fuente.
10. Mezcla:
    `uint8(0.45 * heatmap + 0.55 * source_rgb)`.
11. Clipping y cuantización:
    la combinación de entradas `uint8` y pesos positivos que suman 1 permanece en
    `[0, 255]`; la conversión `np.uint8` trunca. No se añadió clipping distinto.
12. Salida:
    `Image.fromarray(overlay, mode="RGB")`.

No se cambiaron normalización, interpolación, colormap, alpha, fórmula, clipping,
cuantización ni orden de canales. Solo se hizo explícito que el tamaño destino y el
fondo son los de `source_image`.

## Integración con /explain/gradcam

Se mantienen:

- método `POST`;
- ruta `/explain/gradcam`;
- entrada `dataset_id` e `image_id`;
- respuesta `result_id`, `prediction`, `confidence` y `gradcam_image_url`;
- nombre `gradcam.png`;
- URL `/results/{result_id}/gradcam.png`;
- persistencia PNG y, por extensión, MIME servido como PNG;
- prefijo de resultado `gradcam`;
- metadatos, estados y propagación de errores existentes;
- clase objetivo implícita: `argmax(softmax(output[0]))`;
- redondeo de confianza a cuatro decimales.

Predict continúa consumiendo el mismo endpoint y contrato. No se modificó el frontend.

Attack continúa importando y llamando a `generate_gradcam(model, image)` con dos
argumentos. Ese wrapper mantiene su preprocess y su forma de respuesta; no se migró el
Grad-CAM original ni el adversarial del flujo `/attack`.

## Pruebas añadidas

Se añadieron pruebas para:

- fuentes horizontales `(31, 11)`;
- fuentes verticales `(13, 37)`;
- fuentes cuadradas `(23, 23)`;
- dimensiones no múltiplo de 224 `(19, 10)`;
- dimensiones siempre distintas del tensor científico `(7, 5)`;
- tamaño y modo RGB tanto del PIL como del PNG guardado y reabierto;
- uso por identidad del tensor preprocesado y ausencia de una segunda llamada a
  preprocess dentro del cálculo;
- un único forward;
- mezcla exacta `0.45 / 0.55` y orden RGB mediante un colormap controlado;
- inmutabilidad de tamaño y bytes de la fuente;
- eliminación de hooks y gradientes en éxito;
- eliminación de hooks cuando el forward falla;
- compatibilidad del wrapper legacy;
- separación de representaciones en el servicio;
- método, ruta, modelo de respuesta y campos del controlador;
- propagación de errores inesperados del generador.

## Comandos ejecutados y resultados

1. Comando:

   ```text
   python3 -m compileall -q app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py && python3 -m ruff check app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py && python3 -m pytest -q tests/test_gradcam_source_geometry.py tests/test_explainability.py tests/test_unexpected_service_errors.py -k 'gradcam'
   ```

   Resultado: falló antes de Ruff y pytest. `compileall` obtuvo
   `PermissionError: [Errno 13] Permission denied` al intentar escribir bytecode en
   `app/ml/explainability/__pycache__` y `tests/__pycache__`. Ninguna prueba se
   ejecutó en este intento.

2. Comando:

   ```text
   python3 -c "import py_compile; files=['app/ml/explainability/gradcam.py','app/services/explainability_service.py','tests/test_gradcam_source_geometry.py','tests/test_unexpected_service_errors.py']; [py_compile.compile(path, cfile='/tmp/gradcam-subissue-1-'+str(index)+'.pyc', doraise=True) for index, path in enumerate(files)]"
   ```

   Resultado: correcto, exit code 0. Los cuatro archivos compilaron enviando los
   artefactos a `/tmp`.

3. Comando:

   ```text
   python3 -m ruff check app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py
   ```

   Resultado: no ejecutado por el Python del sistema:
   `/usr/bin/python3: No module named ruff`.

4. Comando:

   ```text
   PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_gradcam_source_geometry.py tests/test_explainability.py tests/test_unexpected_service_errors.py -k gradcam
   ```

   Resultado: no ejecutado por el Python del sistema:
   `/usr/bin/python3: No module named pytest`.

5. Comando:

   ```text
   ../venv/bin/ruff check app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py
   ```

   Resultado inicial: 2 incidencias `I001` de orden de imports, 0 pruebas. Fueron
   corregidas de forma mecánica.

6. Comando:

   ```text
   PYTHONDONTWRITEBYTECODE=1 ../venv/bin/pytest -q -p no:cacheprovider tests/test_gradcam_source_geometry.py tests/test_explainability.py tests/test_unexpected_service_errors.py -k gradcam
   ```

   Resultado: el fixture autouse global intentó acceder a MongoDB y quedó esperando
   repetidamente; mostró tres errores de setup (`EEE`) y se interrumpió con exit code
   130. No se contabiliza ninguna prueba como superada en este intento.

7. Comando:

   ```text
   ../venv/bin/ruff check --fix app/services/explainability_service.py tests/test_unexpected_service_errors.py && ../venv/bin/ruff check app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py && PYTHONPATH=app PYTHONDONTWRITEBYTECODE=1 ../venv/bin/pytest -q -p no:cacheprovider --noconftest tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py::test_gradcam_preserves_unexpected_error
   ```

   Resultado: Ruff corrigió 2 imports y después pasó sin incidencias. pytest:
   `7 passed`, `0 failed`, `0 skipped`, con 5 warnings del full backward hook del
   modelo mínimo de prueba.

8. Comando:

   ```text
   ../venv/bin/ruff check app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py && PYTHONPATH=app PYTHONDONTWRITEBYTECODE=1 ../venv/bin/pytest -q -p no:cacheprovider --noconftest tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py::test_gradcam_preserves_unexpected_error
   ```

   Resultado intermedio: Ruff pasó. Las primeras 7 pruebas emitieron puntos, pero la
   prueba que usaba `TestClient` quedó colgada al cerrar el cliente y el proceso se
   interrumpió con exit code 130. Ese intento no se presenta como superado. La prueba
   dependiente se sustituyó por inspección directa de la ruta y llamada async al
   controlador.

9. Comando final:

   ```text
   ../venv/bin/ruff check app/ml/explainability/gradcam.py app/services/explainability_service.py tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py && PYTHONPATH=app PYTHONDONTWRITEBYTECODE=1 ../venv/bin/pytest -q -p no:cacheprovider --noconftest tests/test_gradcam_source_geometry.py tests/test_unexpected_service_errors.py::test_gradcam_preserves_unexpected_error
   ```

   Resultado: Ruff `All checks passed!`. pytest: `9 passed`, `0 failed`,
   `0 skipped`, con 5 warnings del full backward hook del modelo mínimo de prueba.

## Elementos confirmados sin cambios

- Preprocess científico del modelo, incluidos resize y normalización.
- Dispositivo y modo de evaluación del modelo.
- Capa objetivo y tipos de hooks.
- Fórmula Grad-CAM.
- Clase objetivo, predicción y confianza.
- Normalización, interpolación, colormap, alpha, mezcla, clipping, cuantización y RGB.
- Método, ruta, schemas, nombres, URL, formato PNG, estados y errores públicos.
- Persistencia y metadatos.
- FGSM, PGD, Random Noise y reconstrucción visual de ataques.
- `AdversarialAttackResult`.
- Grad-CAM adversarial y endpoint `/attack`.
- Evaluate y detección.
- Frontend.

## Riesgos, limitaciones y trabajo pendiente

- La suite focalizada que carga el `conftest` de integración no pudo completarse porque
  MongoDB no respondió en este entorno. Las pruebas unitarias se ejecutaron con
  `--noconftest`; no se afirma que las pruebas dependientes de Mongo hayan pasado.
- Una prueba HTTP basada en `TestClient` quedó colgada al cerrar el cliente. El contrato
  se comprobó sin servidor inspeccionando método, ruta y `response_model`, y llamando al
  controlador async. No se realizó una petición HTTP de extremo a extremo en el intento
  final.
- PyTorch emitió 5 warnings en el modelo mínimo de prueba porque su entrada directa no
  requería gradiente y el full backward hook se disparó respecto a la salida. No hubo
  fallos y el tipo de hook existente no se cambió.
- El checkout inicial ya usaba el tamaño del PIL fuente en el endpoint limpio; el cambio
  hace explícito y verificable el contrato de separación, pero no permite demostrar una
  diferencia visual de tamaño respecto a ese estado inicial concreto.
- Grad-CAM adversarial y Evaluate quedan expresamente pendientes para sus subissues
  posteriores.

## Desviaciones respecto al prompt

Ninguna.
