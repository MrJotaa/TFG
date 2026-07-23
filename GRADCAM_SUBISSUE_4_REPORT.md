# Grad-CAM Subissue 4 Report

## Alcance implementado

Se ha implementado exclusivamente la exposición en el frontend del Grad-CAM bajo demanda para imágenes de evaluaciones de clasificación. Cada tarjeta de imagen evaluada ofrece una acción explícita para solicitar y mostrar los artefactos Grad-CAM original y adversarial ya soportados por el backend.

No se ha modificado código Python, el pipeline de ataques, el cálculo de evaluaciones, la pantalla Profile ni el flujo de detección.

## Flujo anterior del frontend

La pestaña de imágenes de una evaluación de clasificación mostraba la imagen original, la adversarial, la perturbación y sus diagnósticos. No existía una acción para solicitar los Grad-CAM persistidos por evaluación e imagen ni estado de interfaz asociado a esa operación.

La resolución de URLs de imágenes estaba implementada localmente en `EvaluationImages.jsx`, mientras que el cliente común ya concentraba autenticación y peticiones HTTP.

## Flujo resultante

Tras completar una evaluación de clasificación y abrir la pestaña de imágenes:

1. Cada tarjeta presenta la acción `Generar Grad-CAM`.
2. Solo un clic del usuario dispara la petición para esa evaluación e imagen.
3. La tarjeta pasa de forma independiente por `idle`, `loading`, `success` o `error`.
4. Al completarse, se añaden los Grad-CAM original y adversarial debajo del contenido existente.
5. Un error permite reintentar manualmente desde la misma tarjeta.

## Archivos modificados

- `frontend/src/api/client.js`: resolución compartida de URLs y propagación del estado HTTP en errores.
- `frontend/src/api/evaluation.js`: cliente del endpoint Grad-CAM de evaluación.
- `frontend/src/utils/evaluationGradCamState.js`: claves, reducer, validación y mensajes de error.
- `frontend/src/components/EvaluationPanel.jsx`: estado persistente por imagen y coordinación de las peticiones.
- `frontend/src/components/evaluation/EvaluationImages.jsx`: acción, estados y presentación de los artefactos.
- `frontend/src/index.css`: diseño, estados visuales y adaptación responsive.
- `frontend/tests/evaluationGradCam.test.js`: pruebas focalizadas.
- `frontend/package.json`: script `test:gradcam`.
- `GRADCAM_SUBISSUE_4_REPORT.md`: este informe.

## Integración con el endpoint

Se añadió `generateEvaluationGradCam(evaluationId, imageId)` en el cliente de evaluaciones. Ejecuta:

```text
POST /evaluations/{evaluation_id}/images/{image_id}/gradcam
```

La petición reutiliza `apiRequest`, por lo que conserva la autenticación Bearer existente. No envía cuerpo ni fuerza una cabecera `Content-Type`. Ambos identificadores se codifican con `encodeURIComponent`.

La respuesta se acepta únicamente cuando su `evaluation_id` e `image_id` coinciden con la solicitud y contiene `original_gradcam_image_url` y `adversarial_gradcam_image_url`.

## Identidad de evaluación e imagen

Cada entrada usa una clave compuesta y sin colisiones triviales, construida serializando el par `[evaluation_id, image_id]`. La llamada se realiza con `evaluation.result_id` y `result.image_id`; no se usa el nombre de archivo como identidad.

La validación de la respuesta impide asociar a una tarjeta artefactos pertenecientes a otra evaluación o imagen.

## Estado por imagen

El estado vive en `EvaluationPanel`, por encima de la pestaña paginada, y se indexa mediante la clave compuesta. Esto permite que cada imagen mantenga de forma independiente:

- `idle`: acción disponible.
- `loading`: petición en curso y botón bloqueado.
- `success`: URLs almacenadas y artefactos visibles.
- `error`: mensaje visible y reintento manual habilitado.

Al iniciar una evaluación nueva se limpia el estado completo. Un contador de ejecución descarta también respuestas tardías de una evaluación anterior. Un registro síncrono de peticiones en curso evita dobles envíos antes de que React procese el siguiente render.

## Generación bajo demanda

La función de red solo se invoca desde el manejador conectado al `onClick` del botón de una tarjeta. No hay efectos de montaje, precarga, generación al cambiar de página o pestaña, peticiones por hover ni acción masiva.

Una entrada en `loading` o `success` no puede volver a solicitarse. Una entrada en `error` solo se reintenta mediante un nuevo clic.

## Presentación de Grad-CAM

Los dos resultados se incorporan como contenido adicional:

- `Grad-CAM original`.
- `Grad-CAM adversarial`.

No reemplazan ni ocultan la imagen original, la adversarial o la perturbación. Las URLs relativas se resuelven con la base de la API y las absolutas se conservan sin duplicar host.

## Geometría y diseño responsive

Los Grad-CAM se presentan en una cuadrícula de dos columnas en espacio suficiente y en una sola columna en pantallas estrechas. Las imágenes usan el ancho disponible, altura automática y `object-fit: contain`, evitando recortes y conservando su proporción natural.

La zona de acción adapta su distribución en móvil y el botón ocupa el ancho disponible bajo el breakpoint existente de 760 px.

## Manejo de errores

`apiRequest` conserva ahora `response.status` en el objeto `Error`. La interfaz traduce de forma controlada:

- `400`: evaluación incompleta, datos inválidos o modelo no coincidente.
- `401`: sesión caducada.
- `403`: falta de permiso.
- `404`: evaluación o imagen inexistente.
- `410`: artefactos temporales no disponibles e indicación explícita de repetir la evaluación.
- `500`: fallo de generación o persistencia.
- Error de red u otro estado: fallo de conexión comprensible.

Los errores se limitan a la tarjeta afectada y no invalidan resultados obtenidos por otras imágenes.

## Accesibilidad

La acción es un elemento `button` real, con texto de estado visible y bloqueo nativo durante carga y tras éxito. La carga usa `role="status"` con `aria-live="polite"` y el error usa `role="alert"`; el botón se relaciona con el error mediante `aria-describedby`.

La sección tiene una etiqueta contextual basada en el nombre de archivo. Ambos artefactos incluyen etiquetas y textos alternativos descriptivos, y el foco visible reutiliza el tratamiento accesible existente para botones del panel.

## Clasificación y detección

La funcionalidad se añadió únicamente a `EvaluationPanel` y `EvaluationImages`, que componen el flujo de clasificación. `EvaluationView` continúa seleccionando por separado `DetectionEvaluationPanel` cuando el modelo activo es de detección.

No se añadió ninguna acción Grad-CAM al componente de detección ni se modificó su comportamiento.

## Peticiones y ejecuciones observables

Por cada clic válido se observa una única petición `POST` autenticada, sin body, a la evaluación e imagen exactas. El registro en memoria bloquea un segundo clic concurrente incluso antes de que se refleje `loading`.

No se ejecuta una evaluación, ataque o inferencia adicional desde esta integración. El frontend se limita a solicitar el endpoint Grad-CAM bajo demanda y presentar las dos URLs devueltas.

## Pruebas añadidas

Se añadió una suite focalizada con el runner nativo de Node, sin introducir dependencias:

- método, ruta, autenticación y ausencia de body;
- aislamiento por evaluación e imagen y limpieza del reducer;
- prevención de duplicados y habilitación del reintento;
- validación de identidad y presencia de ambas URLs;
- resolución de URLs relativas, absolutas y nulas;
- mensajes para `400`, `401`, `403`, `404`, `410`, `500` y red.

El proyecto no dispone de un runner de componentes React ni de Testing Library. Por ello, el comportamiento JSX/CSS se validó mediante lint y compilación de producción, mientras que la lógica observable se extrajo y cubrió con pruebas unitarias.

## Comandos ejecutados y resultados

Todos los comandos se ejecutaron sin Git. Los cinco primeros se lanzaron desde la raíz del repositorio:

```text
sed -n '1,820p' '/home/mrj/.codex/attachments/56383147-c92f-4b03-bc01-3925aa22272b/pasted-text.txt'; find .. -name AGENTS.md -print; ls -la GRADCAM_SUBISSUE_*_REPORT.md 2>/dev/null || true
```

Resultado: se leyó el prompt completo; solo se encontró un `AGENTS.md` dentro de `frontend/node_modules/recharts`, fuera del alcance; el glob de informes no encontró coincidencias visibles.

```text
sed -n '1,260p' frontend/package.json; find frontend/src -maxdepth 4 -type f | sort; sed -n '1,260p' frontend/src/api/client.js; sed -n '1,180p' frontend/src/api/evaluation.js; rg -n "API_BASE_URL|resolve.*Url|image.*url|evaluation|Evaluate|pagination|page|filter|diagnostic" frontend/src -g '*.js' -g '*.jsx' -g '*.css'
```

Resultado: inspección correcta de scripts, cliente API y estructura del frontend.

```text
sed -n '1,460p' frontend/src/components/evaluation/EvaluationImages.jsx; sed -n '1,250p' frontend/src/components/EvaluationPanel.jsx; sed -n '1230,1780p' frontend/src/index.css; sed -n '1780,2380p' frontend/src/index.css; sed -n '1,180p' frontend/eslint.config.js; sed -n '1,120p' frontend/vite.config.js; sed -n '1,120p' frontend/src/context/AuthContext.jsx; sed -n '1,120p' frontend/src/components/DetectionEvaluationPanel.jsx
```

Resultado: se confirmó la separación clasificación/detección, el wrapper autenticado y la ausencia de infraestructura de tests React.

```text
rg -n "evaluation-image-(card|triptych|preview|summary|diagnostics|detail|list|toolbar)|evaluation-images-tab|evaluation-pagination" frontend/src/index.css; sed -n '2570,3000p' frontend/src/index.css; sed -n '140,235p' frontend/src/components/EvaluationPanel.jsx; sed -n '1,120p' frontend/src/pages/EvaluationView.jsx; sed -n '1,100p' frontend/src/pages/Dashboard.jsx; rg -n "EvaluationPanel|DetectionEvaluationPanel|task_type|classification|detection" frontend/src/pages frontend/src/components -g '*.jsx' | sed -n '1,180p'
```

Resultado: se localizaron tarjetas, paginación, responsive y el selector por tipo de tarea.

```text
sed -n '3240,3605p' frontend/src/index.css; sed -n '3605,3795p' frontend/src/index.css; sed -n '1,225p' frontend/src/components/EvaluationPanel.jsx; sed -n '105,140p' frontend/src/pages/EvaluationView.jsx
```

Resultado: se verificó que el estado debía residir en el panel padre para sobrevivir al desmontaje de pestaña y a la paginación.

Los siguientes comandos se lanzaron desde `frontend/`:

```text
./node_modules/.bin/eslint src/api/client.js src/api/evaluation.js src/utils/evaluationGradCamState.js src/components/EvaluationPanel.jsx src/components/evaluation/EvaluationImages.jsx tests/evaluationGradCam.test.js
```

Primera y última ejecución: exit code 0, sin avisos.

```text
npm run test:gradcam
```

Primera ejecución: exit code 1 porque Node ESM no resolvía el import interno `./client` sin extensión. Se corrigió a `./client.js`. Segunda ejecución: exit code 0. Ejecución final tras el refuerzo contra respuestas tardías: exit code 0, 1 archivo de pruebas aprobado y 0 fallos.

```text
npm run build
```

Resultado: exit code 0; Vite 8.1.3 transformó 631 módulos y generó la distribución en 570 ms. Se mantiene el aviso preexistente de un chunk JavaScript superior a 500 kB (`731.35 kB`, `208.39 kB` gzip).

```text
sed -n '1,220p' src/utils/evaluationGradCamState.js; sed -n '1,240p' tests/evaluationGradCam.test.js; rg -n "generateEvaluationGradCam|gradCam|Grad-CAM|resolveApiUrl" src/components/EvaluationPanel.jsx src/components/evaluation/EvaluationImages.jsx src/api/client.js src/api/evaluation.js src/index.css
```

Resultado: revisión correcta de lógica, cobertura y puntos de integración.

```text
sed -n '1,175p' src/components/EvaluationPanel.jsx; sed -n '190,285p' src/components/evaluation/EvaluationImages.jsx; sed -n '1,90p' src/api/client.js; sed -n '1,60p' src/api/evaluation.js
```

Resultado: revisión final del manejador, renderizado y contrato del cliente.

No se ejecutaron Ruff, pytest ni otros tests de backend porque no hubo cambios Python.

## Elementos confirmados sin cambios

- Endpoint y servicios del backend.
- Implementación de Grad-CAM del modelo.
- Generación y parámetros de ataques.
- Inicio y resultados de la evaluación base.
- Componentes y comportamiento de detección.
- Profile.
- Imágenes original, adversarial y perturbación ya existentes.
- Filtros, paginación y diagnósticos de la pestaña de imágenes.

## Riesgos, limitaciones y trabajo pendiente

- No hay prueba DOM del componente porque el repositorio no incluye runner React ni Testing Library; añadir esa infraestructura excedería esta subissue. La lógica está cubierta con Node y el JSX/CSS compiló correctamente.
- El build conserva un aviso de tamaño de chunk que ya afecta al bundle general y no es específico de Grad-CAM.
- La disponibilidad real de los artefactos sigue dependiendo del ciclo de vida temporal del backend; el estado `410` informa al usuario de que debe repetir la evaluación.
- No se realizó prueba E2E contra un backend y modelo reales en este entorno.

## Desviaciones respecto al prompt

No hay desviaciones funcionales conocidas. La validación automatizada de interfaz se realizó con pruebas unitarias de lógica, lint acotado y build, en lugar de una prueba DOM, porque el proyecto no tiene infraestructura de tests de componentes y no se añadieron dependencias fuera del alcance.
