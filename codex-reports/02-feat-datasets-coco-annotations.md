# Informe: anotaciones COCO y estados de revisión

## 1. Estado

- Resultado: completado, incluidas las tres correcciones de concurrencia
  encontradas durante las revisiones previas al commit.
- Resumen: se incorporó un esquema canónico para categorías y objetos,
  importación COCO JSON como snapshot lógico atómico, migración
  aditiva/reversible, cobertura protegida por propietario y una interfaz mínima
  integrada en la gestión actual de datasets. La auditoría posterior detectó
  que la limpieza general de snapshots inactivos podía borrar otra importación
  todavía en preparación; esta iteración limita la limpieza al snapshot exacto
  observado y sustituido. La revisión final detectó además que una respuesta de
  importación podía releer su snapshot después de que otra operación lo
  sustituyera y limpiara; ahora cada respuesta se deriva de los datos canónicos
  estables de su propia operación. Una revisión adicional encontró que el
  `GET` podía mezclar el documento de A con categorías/objetos retirados durante
  la consulta; ahora verifica optimistamente revisión y snapshot y reintenta
  sobre el estado nuevo.
- Hash de `HEAD` inicial:
  `dee4f66e6267bcff747e8b620a04124fbd1ba838`.
- Al comenzar esta iteración, la implementación principal COCO seguía sin
  commit: había 12 archivos modificados y 12 archivos no rastreados. Todos esos
  cambios previos se conservaron. Esta corrección residual también se limitó al
  servicio de anotaciones, sus tests y este informe.
- No se hizo commit ni push.

## 2. Arquitectura inicial

### Modelo de datasets e imágenes

El proyecto no usa ORM ni Beanie. Accede directamente a MongoDB mediante
`pymongo.AsyncMongoClient`.

- Colección `datasets`:
  - `_id` de MongoDB;
  - `name`;
  - `owner_id`;
  - `image_count`;
  - `path`;
  - array embebido `images`.
- Cada imagen embebida contenía:
  - UUID interno `id`;
  - `filename` original de la subida;
  - `stored_filename` aleatorio basado en el UUID;
  - `path` físico confiable;
  - etiqueta libre `label`.
- La colección `users` mantiene referencias a IDs de datasets.

No existían categorías de detección, objetos, cajas, dimensiones persistidas,
estado de revisión ni snapshots.

### Flujo de creación, consulta y eliminación

`backend/app/api/routes/datasets.py` delega en
`backend/app/services/dataset_service.py` para:

- crear un dataset y guardar sus imágenes;
- listar datasets e imágenes;
- añadir/eliminar imágenes;
- editar la etiqueta libre;
- eliminar un dataset y su directorio físico.

Las imágenes se guardan bajo:

```text
uploads/datasets/user_<user_id>/dataset_<dataset_id>/<uuid>.<ext>
```

El nombre original se preservaba como `filename`, mientras que el path físico
no dependía de ese nombre. Esta separación permite asociar COCO de forma segura
sin abrir nunca `images[].file_name` como una ruta del sistema.

### Autenticación y propiedad

El proyecto dispone de JWT, `require_current_user` y errores `401/403`, pero
los endpoints legacy de datasets no exigen autenticación. Para no romper ese
contrato fuera de alcance, los dos endpoints nuevos sí exigen JWT y validan que
`dataset.owner_id == current_user.id`.

### Frontend relevante

- `DatasetManager.jsx` es la vista existente de gestión.
- `ImageGallery.jsx` ya lista las imágenes.
- `frontend/src/api/datasets.js` centraliza las llamadas.
- `apiRequest` ya adjunta el bearer token almacenado.

La nueva funcionalidad se integró en esos puntos, sin crear otra pantalla.

### Limitaciones detectadas

- No había framework de migraciones.
- MongoDB se ejecuta como instancia standalone en Docker Compose; no puede
  asumirse soporte de transacciones multi-documento de replica set.
- El lint global del frontend ya fallaba por reglas React en archivos no
  relacionados.
- Los tests eliminan la base entre casos; los índices deben recrearse de forma
  explícita en los tests que los verifican.

## 3. Diseño implementado

### Modelo canónico

```text
Dataset
 ├─ images[]                         (embebidas)
 │   ├─ id
 │   ├─ original_file_name
 │   ├─ width / height
 │   └─ annotation_review_status
 ├─ active_annotation_snapshot_id ───────────────┐
 └─ dataset_revision                              │
                                                  │
annotation_categories                             │
 ├─ _id (identificador interno)                   │
 ├─ dataset_id                                    │
 ├─ snapshot_id ◄─────────────────────────────────┤
 ├─ coco_category_id                              │
 ├─ name                                          │
 └─ supercategory                                 │
                                                  │
object_annotations                                │
 ├─ _id (identificador interno)                   │
 ├─ dataset_id                                    │
 ├─ snapshot_id ◄─────────────────────────────────┘
 ├─ image_id
 ├─ category_id
 ├─ coco_annotation_id
 ├─ bbox_x1 / bbox_y1 / bbox_x2 / bbox_y2
 ├─ area
 └─ iscrowd
```

Las colecciones nuevas son canónicas: un consumidor futuro no necesita conocer
COCO JSON. El importador entrega `CanonicalAnnotationSnapshot`,
`CanonicalCategory` y `CanonicalObjectAnnotation`.

### Revisión, identidad de snapshot y contrato HTTP

Estos conceptos tienen responsabilidades distintas:

- `dataset_revision` es un contador interno para control de concurrencia
  optimista. El CAS exige el valor observado y lo incrementa al activar.
- `snapshot_id` identifica los documentos de una importación preparada;
  `active_annotation_snapshot_id` selecciona qué identidad es visible.
- No existe un campo de versión del contrato HTTP. Ninguno de los dos
  conceptos anteriores versiona la API, y esta corrección no añade
  `schema_version` ni cambia request/response alguno.

### Relaciones, índices y restricciones

Índices añadidos:

- `datasets.owner_id`;
- `datasets.active_annotation_snapshot_id`;
- categoría única por
  `(dataset_id, snapshot_id, coco_category_id)`;
- nombre de categoría único por `(dataset_id, snapshot_id, name)`;
- anotación única por
  `(dataset_id, snapshot_id, coco_annotation_id)`;
- consulta de objetos por
  `(dataset_id, snapshot_id, image_id)`;
- consulta de objetos por
  `(dataset_id, snapshot_id, category_id)`.

El servicio valida referencias antes de insertar. Las categorías y objetos se
insertan con `insert_many`, no por fila.

### Bounding boxes

COCO `[x, y, width, height]` se convierte sin redondeo a:

```text
x1 = x
y1 = y
x2 = x + width
y2 = y + height
```

Se almacenan `float` BSON de doble precisión, absolutos y no normalizados. Se
rechazan:

- valores no numéricos o booleanos;
- `NaN`, infinitos y números que desborden `float`;
- ancho/alto de caja `<= 0`;
- `x < 0` o `y < 0`;
- `x2 > image.width` o `y2 > image.height`.

No se recortan ni corrigen cajas silenciosamente.

### Dimensiones autoritativas

Las nuevas imágenes guardan `width` y `height` obtenidos con Pillow desde el
archivo físico. Para imágenes legacy sin dimensiones, la importación abre el
archivo confiable una sola vez, lee sus dimensiones y las persiste al activar
el snapshot. El JSON debe declarar dimensiones enteras positivas que coincidan
exactamente con las dimensiones reales.

### Estados e invariantes

Se almacena un enum estable por imagen:

| Estado | Revisada | Objetos |
|---|---:|---:|
| `unreviewed` | No | 0 |
| `reviewed_empty` | Sí | 0 |
| `reviewed_with_objects` | Sí | >= 1 |

El servicio calcula todos los estados a partir del snapshot validado y los
activa en la misma actualización atómica que cambia el puntero del snapshot.
No hay endpoints para escribir manualmente el estado.

### Agregación de cobertura compartida

`_build_annotation_coverage` es una función pura que construye
`DatasetAnnotationCoverage` a partir de:

- imágenes con sus estados ya resueltos;
- categorías canónicas;
- recuentos de objetos por identidad interna de categoría;
- presencia o ausencia de snapshot.

La importación la alimenta con `updated_images`, los documentos de categoría
preparados desde `CanonicalAnnotationSnapshot` y recuentos obtenidos de sus
anotaciones validadas. Esos valores pertenecen exclusivamente a esa operación
y permanecen disponibles aunque sus documentos Mongo sean sustituidos.

La consulta pública mantiene otra fuente: lee el dataset y las
categorías/recuentos del `active_annotation_snapshot_id` actual y pasa esos
resultados por la misma función. Así cambia la fuente según el propósito, pero
no se duplica la semántica de estados, porcentajes, categorías ni totales.

### Lectura optimista consistente del `GET`

`get_annotation_coverage` ejecuta como máximo
`ANNOTATION_COVERAGE_MAX_READ_ATTEMPTS = 3` intentos:

1. conserva `dataset_revision` y `active_annotation_snapshot_id` del documento
   observado;
2. calcula categorías, objetos y estados para ese snapshot;
3. relee el dataset y vuelve a validar existencia y propietario;
4. devuelve solo si revisión y snapshot siguen siendo idénticos;
5. si cambiaron, reutiliza el documento recién leído como base del intento
   siguiente.

La segunda lectura incluye las imágenes, por lo que un reintento no necesita
una tercera lectura de preparación. Si el dataset desaparece o cambia de
propietario se conservan `404`/`403`; si los tres intentos encuentran cambios,
se usa el `ConflictError` existente (`409`) con indicación de reintento.

### Límite de importadores

`services/annotation_importers/base.py` define el protocolo y las entidades
canónicas. `registry.py` resuelve el formato. COCO vive exclusivamente en
`coco.py`.

Un futuro importador YOLO TXT podrá producir el mismo
`CanonicalAnnotationSnapshot` sin modificar las colecciones, la cobertura ni
los futuros consumidores de métricas.

## 4. Semántica de importación

### Proceso completo

1. Se valida JWT y propiedad del dataset.
2. Se valida nombre, MIME y límite del archivo.
3. El JSON se lee por chunks hasta un máximo configurable.
4. Se parsea y valida por completo.
5. Se asocian imágenes y se contrastan dimensiones/cajas.
6. Se construyen el snapshot canónico, los estados de imagen y su cobertura
   estable en memoria.
7. Se insertan categorías y objetos por lotes con un `snapshot_id` nuevo.
   Todavía son invisibles para consumidores.
8. Una única actualización de `datasets`:
   - exige el `dataset_revision` y snapshot anterior observados;
   - actualiza todos los estados/dimensiones;
   - cambia `active_annotation_snapshot_id`;
   - incrementa `dataset_revision`.
9. Si el control optimista falla, se elimina lo preparado y se responde `409`
   con `snapshot_applied: false`.
10. Tras activar, solo se limpia el `old_snapshot_id` que esa operación observó
    y sustituyó. Si era nulo, no se ejecuta limpieza posterior.
11. Se devuelve la cobertura estable de esa importación, sin releer sus
    documentos persistidos.

Toda la validación sucede antes de escribir. MongoDB standalone no ofrece la
transacción multi-documento requerida sin replica set; el patrón
stage-and-switch proporciona atomicidad lógica observable y evita depender de
una configuración no soportada por el proyecto.

### Incidencia de concurrencia encontrada y corregida

La revisión previa al commit encontró que el paso 10 eliminaba categorías y
objetos del dataset cuyo `snapshot_id` fuese distinto del recién activado. Esa
limpieza general era insegura aunque el CAS de activación fuese correcto:

1. A activaba su snapshot.
2. B observaba A como activo y preparaba sus documentos.
3. La limpieza posterior de A podía eliminar la preparación de B.
4. B superaba su CAS y activaba un puntero a documentos que ya no existían.

El CAS no era el defecto: ya compara propietario, `dataset_revision` observado
y `active_annotation_snapshot_id` observado, y se conserva sin cambios. El
problema era que una eliminación posterior actuaba sobre IDs que A nunca había
observado.

La corrección usa `_cleanup_snapshot(object_id, old_snapshot_id)` únicamente
cuando `old_snapshot_id` no es nulo. Por tanto, A solo puede eliminar el
snapshot que efectivamente sustituyó y no puede alcanzar el ID nuevo de B. No
se incorporó un recolector general:

- tras éxito se limpia exclusivamente el snapshot anterior conocido;
- si el CAS falla, el bloque de rollback limpia exclusivamente el `snapshot_id`
  preparado por la operación perdedora;
- si una escritura falla tras insertar categorías, el mismo rollback elimina
  únicamente ese snapshot parcialmente preparado;
- el snapshot activo previo y otras preparaciones quedan fuera de todos esos
  filtros exactos.

La limpieza posterior a una activación sigue siendo best-effort: si MongoDB
falla, el snapshot nuevo ya está activo y el anterior puede quedar huérfano
pero invisible. Una futura recolección segura de huérfanos requeriría distinguir
preparaciones en curso y queda expresamente fuera de esta sub-issue.

### Carrera residual de la respuesta encontrada y corregida

La revisión final previa al commit detectó un defecto posterior e independiente
de la integridad persistente. Después de que A activase correctamente
`snapshot_A`, construía su respuesta llamando a
`_coverage_for_dataset_document` con el ID de A. En ese intervalo podía ocurrir:

1. A activa `snapshot_A` y todavía no termina su respuesta.
2. B observa A, prepara y activa `snapshot_B`.
3. B limpia correctamente `snapshot_A`, porque es el ID anterior exacto que
   sustituyó.
4. A consulta categorías y objetos de `snapshot_A`, ya inexistentes, y puede
   responder con cero categorías/objetos.

El dataset nunca apuntaba a documentos ausentes: al limpiar A, B ya era el
snapshot activo. El CAS, la revisión, el puntero y la limpieza exacta protegían
la persistencia; lo incoherente era únicamente la respuesta tardía de A.

Ahora `import_coco_annotations` prepara `import_coverage` antes de cualquier
escritura mediante `_build_annotation_coverage`. Su fuente son
`updated_images`, `category_documents` derivados de categorías canónicas y los
recuentos por categoría calculados desde las anotaciones del snapshot ya
validado. Tras un CAS exitoso devuelve esa estructura inmutable desde la
perspectiva de la operación; no vuelve a consultar categorías u objetos.

Por tanto:

- `POST .../annotations/coco` describe siempre el contenido validado y activado
  por esa llamada, aunque una llamada posterior ya lo haya sustituido;
- `GET .../annotations/coverage` sigue leyendo exclusivamente el snapshot
  activo en el momento de la consulta;
- ambos caminos comparten `_build_annotation_coverage`, por lo que estados,
  porcentaje, distribución y totales no divergen.

### Carrera residual del `GET` encontrada y corregida

La lectura pública aún podía observar A, cargar parte de sus categorías y,
mientras agregaba objetos, permitir que B se activase y limpiase A. El cálculo
podía mezclar `images`/estados de A con categorías u objetos ausentes. A
diferencia del defecto del `POST`, aquí no existe una representación histórica
propia que deba devolverse: el contrato del `GET` es describir el estado activo.

La solución aplica validación optimista alrededor de la lectura:

1. captura `(dataset_revision, active_annotation_snapshot_id)`;
2. calcula toda la cobertura para esa pareja;
3. relee ambos valores;
4. descarta por completo el resultado si cambió cualquiera;
5. repite desde el documento nuevo, hasta tres intentos.

No se fusionan resultados entre intentos. La prueba detiene el primer agregado
de A, activa B y confirma que B elimina A; al continuar, el servicio detecta el
cambio, ejecuta una ronda completa sobre B y devuelve exactamente la misma
cobertura que produjo la importación B. No se modificaron
`import_coco_annotations`, su `import_coverage`, el CAS ni las limpiezas.

### Política snapshot

- Una importación correcta reemplaza el snapshot completo.
- Imágenes incluidas con objetos: `reviewed_with_objects`.
- Imágenes incluidas sin objetos: `reviewed_empty`.
- Imágenes ausentes: `unreviewed`.
- Reimportar el mismo JSON produce el mismo estado y solo un snapshot activo;
  no duplica objetos o categorías activas.
- Las etiquetas libres no participan ni se reinterpretan.

### Asociación exacta de imágenes

1. Se normalizan separadores `\` a `/`.
2. Se rechazan rutas absolutas, unidades Windows, segmentos `.`/`..`,
   controles y nombres vacíos.
3. Se busca coincidencia exacta contra `original_file_name` normalizado,
   conservando directorios.
4. Solo si no existe coincidencia exacta, se usa basename.
5. El fallback se acepta únicamente si identifica una sola imagen.
6. Cero o varias candidatas abortan la importación.
7. Dos imágenes COCO no pueden asociarse a la misma imagen del dataset.

`file_name` nunca se concatena con rutas ni se abre.

### Política de errores y rollback

Los errores de validación son `400` y contienen:

- `code`;
- `message`;
- `section`;
- `field`, cuando aplica;
- `coco_id`, cuando aplica;
- `image_name`, cuando aplica;
- `snapshot_applied: false`.

Un conflicto concurrente devuelve `409` y confirma también que no se activó el
snapshot. Los errores de validación no eliminan ni modifican el snapshot
anterior. Un error de persistencia durante la preparación se propaga conforme
al manejo existente de errores inesperados, después de intentar borrar por ID
exacto los documentos parciales de esa operación; no avanza la revisión ni
cambia el puntero activo.

### Límite de subida

`COCO_MAX_UPLOAD_BYTES` es configurable por entorno y vale por defecto
`10 MiB` (`10 * 1024 * 1024`). Se usa un `bytearray` incremental de 64 KiB para
evitar copias acumulativas. Se aceptan `.json` con MIME JSON u octet-stream.

### Ejemplo COCO aceptado

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "train/cat.png",
      "width": 640,
      "height": 480
    },
    {
      "id": 2,
      "file_name": "train/empty.png",
      "width": 640,
      "height": 480
    }
  ],
  "categories": [
    {
      "id": 7,
      "name": "cat",
      "supercategory": "animal"
    }
  ],
  "annotations": [
    {
      "id": 101,
      "image_id": 1,
      "category_id": 7,
      "bbox": [10.25, 20.5, 100.125, 80.75],
      "area": 8085.09375,
      "iscrowd": 0
    }
  ]
}
```

### Respuesta HTTP de éxito

```json
{
  "dataset_id": "507f1f77bcf86cd799439011",
  "format": "coco",
  "replaced_existing_snapshot": false,
  "coverage": {
    "dataset_id": "507f1f77bcf86cd799439011",
    "has_annotation_snapshot": true,
    "total_images": 3,
    "reviewed_images": 2,
    "unreviewed_images": 1,
    "reviewed_empty_images": 1,
    "reviewed_with_objects_images": 1,
    "coverage_percentage": 66.67,
    "total_objects": 1,
    "total_categories": 1,
    "objects_by_category": [
      {
        "category_id": "internal-category-uuid",
        "coco_category_id": 7,
        "name": "cat",
        "object_count": 1
      }
    ]
  }
}
```

### Respuesta HTTP de error

```json
{
  "detail": {
    "code": "invalid_coco_annotations",
    "message": "Bounding box is outside authoritative image bounds",
    "section": "annotations",
    "field": "bbox",
    "coco_id": 101,
    "image_name": "train/cat.png",
    "snapshot_applied": false
  }
}
```

## 5. Migración y compatibilidad

### Migración creada

`backend/app/db/migrations.py` contiene:

- `upgrade_coco_annotations_schema`;
- `downgrade_coco_annotations_schema`.

La migración se ejecuta de forma idempotente en el lifespan de FastAPI y puede
recibir una base inyectada para pruebas/operación.

El upgrade:

- añade `dataset_revision = 0` si falta;
- añade `annotation_review_status = "unreviewed"` a imágenes existentes;
- copia `filename` a `original_file_name` si falta;
- conserva campos ya migrados mediante `$mergeObjects`;
- crea índices.

El downgrade:

- elimina colecciones canónicas;
- elimina los campos añadidos de imágenes/dataset;
- conserva imágenes, nombres, paths y etiquetas.

### Compatibilidad

- Todos los campos públicos anteriores permanecen.
- `width`, `height` y `annotation_review_status` son aditivos.
- Datasets de clasificación siguen usando `label` sin cambios.
- No se convierte ninguna etiqueta libre en categoría u objeto.
- No se exige reconstruir datasets.
- Añadir imágenes crea `unreviewed`.
- Eliminar una imagen borra sus objetos en todos los snapshots.
- Eliminar un dataset borra categorías y objetos asociados.
- La suite completa de clasificación y backend pasa.

## 6. Cambios realizados

### Migraciones y modelos

- `backend/app/db/collections.py`
  - Añade `annotation_categories_collection` y
    `object_annotations_collection`.
- `backend/app/db/migrations.py`
  - Añade upgrade/downgrade idempotentes, índices y migración de imágenes.
- `backend/app/main.py`
  - Añade lifespan que garantiza la migración al iniciar.
- `backend/app/schemas/datasets.py`
  - Añade `AnnotationReviewStatus`, dimensiones, estado público,
    `CategoryObjectCount`, `DatasetAnnotationCoverage` y
    `CocoAnnotationImportResponse`.
- `backend/app/core/errors.py`
  - Permite detalles estructurados y añade `ConflictError`.

### Servicios e importadores

- `backend/app/services/annotation_importers/__init__.py`
  - Exporta el registro de importadores.
- `backend/app/services/annotation_importers/base.py`
  - Define protocolo, identidad segura y entidades canónicas.
- `backend/app/services/annotation_importers/registry.py`
  - Resuelve importadores por formato sin acoplar API/servicios a COCO.
- `backend/app/services/annotation_importers/coco.py`
  - Parseo, asociación y validación exhaustiva; conversión `xywh -> xyxy`.
- `backend/app/services/dataset_annotation_service.py`
  - Autorización, límite de upload, snapshot stage-and-switch, cobertura,
    control optimista, limpieza por el ID exacto observado/sustituido y
    agregación pura compartida para respuestas estables; el `GET` valida
    optimistamente revisión/snapshot con reintentos acotados.
- `backend/app/services/dataset_service.py`
  - Persiste identidad/dimensiones/estado en nuevas imágenes, incrementa
    revisión ante cambios de imagen/label y limpia entidades al eliminar.

### API y schemas

- `backend/app/api/routes/datasets.py`
  - Añade:
    - `GET /datasets/{dataset_id}/annotations/coverage`;
    - `POST /datasets/{dataset_id}/annotations/coco`.
  - Ambos requieren JWT y propiedad.

### Frontend

- `frontend/src/api/client.js`
  - Conserva errores estructurados en `error.details`.
- `frontend/src/api/datasets.js`
  - Añade consulta de cobertura e importación multipart.
- `frontend/src/utils/cocoAnnotations.js`
  - Etiquetas estables, confirmación, bloqueo, actualización y errores.
- `frontend/src/components/CocoAnnotationPanel.jsx`
  - Selector `.json`, carga, éxito/error, confirmación, cobertura y
    distribución; exporta `CoverageSummary` puro para SSR.
- `frontend/src/pages/DatasetManager.jsx`
  - Integra el panel y refresca imágenes tras importar.
- `frontend/src/components/ImageGallery.jsx`
  - Presenta el estado de revisión por imagen.
- `frontend/src/index.css`
  - Estilos del panel, distribución y estados.
- `frontend/package.json`
  - Añade `test:coco`.

### Tests

- `backend/tests/test_coco_annotation_importer.py`
  - 21 casos del importador canónico y validaciones.
- `backend/tests/test_dataset_coco_annotations.py`
  - 20 casos de migración, DB, API, seguridad, snapshot, cascadas, agregación
    estable, lectura optimista y regresiones deterministas de concurrencia/
    escritura parcial.
- `frontend/tests/cocoAnnotations.test.js`
  - SSR real del resumen y pruebas de estado, reemplazo, actualización,
    errores y bloqueo.

### Documentación

- `codex-reports/02-feat-datasets-coco-annotations.md`
  - Este informe.

## 7. Eficiencia y escalabilidad

### Estrategia transaccional

- Validación completa antes de escribir.
- Snapshot nuevo preparado con ID independiente.
- Un único `update_one` activa puntero, estados y dimensiones.
- El CAS combina propietario, `dataset_revision` y snapshot activo observados;
  evita sobrescribir altas, bajas, cambios o importaciones concurrentes.
- Un perdedor del CAS y una preparación fallida limpian su propio ID.
- Un ganador limpia solo el ID anterior que sustituyó; nunca consulta por
  desigualdad ni barre otros IDs inactivos.
- La respuesta exitosa queda ligada a los datos validados de su operación y no
  necesita prolongar la vida del snapshot ni bloquear importaciones posteriores.
- Los lectores solo deben consultar el snapshot activo.

### Operaciones por lotes

- Un `insert_many` para categorías.
- Un `insert_many` para objetos.
- Un `update_one` para activar.
- Si existía snapshot anterior, un `delete_many` por colección dirigido
  exclusivamente a ese `old_snapshot_id`.
- Eliminaciones por dataset/imagen con `delete_many`.

No hay commits/escrituras por objeto.

### Cobertura

La respuesta de importación no añade consultas: agrega en `O(images +
categories + annotations)` sobre estructuras ya construidas para validar,
persistir y activar.

Sin concurrencia, el endpoint público usa una sola ronda y cuatro consultas:

1. documento del dataset;
2. categorías del snapshot;
3. agregación `$group` de objetos por categoría.
4. verificación final del documento del dataset.

Si debe reintentar, reutiliza la verificación como documento inicial y añade
solo categorías, agregación y nueva verificación por ronda. El máximo de tres
intentos acota el coste. No ejecuta consultas por imagen ni categoría. Ambos
caminos usan la misma función pura para ensamblar el schema final.

### Memoria e imágenes

- JSON limitado a 10 MiB y leído incrementalmente.
- Se mantiene una sola representación decodificada del JSON y su snapshot.
- No se cargan píxeles de imágenes en memoria.
- Pillow abre cada imagen sin dimensiones persistidas una sola vez para leer
  cabecera/tamaño.

### Costes conocidos

- La validación mantiene metadatos COCO y snapshot canónico en memoria:
  complejidad `O(images + categories + annotations)`.
- Las colecciones separadas evitan el límite BSON de 16 MiB que implicaría
  embebir todos los objetos en el dataset.
- El límite de 10 MiB puede ajustarse en despliegues con COCO mayores.

## 8. Decisiones tomadas

### Colecciones separadas e identidad de snapshot

- Decisión: categorías/objetos separados por `snapshot_id`.
- Alternativas: embebidos en dataset o transacción multi-documento.
- Motivo: los embebidos no escalan por el límite BSON; las transacciones
  requieren replica set ausente en la configuración soportada.
- Impacto: futuras métricas consultarán por dataset/snapshot con índices.

### Atomicidad lógica stage-and-switch

- Decisión: preparar filas invisibles y activar mediante un puntero atómico.
- Alternativas: borrar e insertar en sitio o exigir replica set.
- Motivo: nunca deja al lector un snapshot parcial y funciona en Mongo
  standalone.
- Impacto: un apagado abrupto antes de activar puede dejar filas inactivas,
  pero no altera el estado visible. No se eliminan de forma oportunista en otra
  importación porque podrían pertenecer a una preparación concurrente; un
  recolector seguro queda como mejora independiente.

### Respuesta de importación desde datos estables

- Decisión: construir la cobertura de `POST` desde el snapshot canónico y el
  plan preparado por esa operación.
- Alternativas: releer Mongo, retrasar la limpieza o serializar imports con un
  lock global.
- Motivo: una relectura puede perder sus documentos tras una sustitución
  legítima; retrasar limpieza no elimina todos los intercalados y un lock
  reduce concurrencia sin proteger mejor la persistencia.
- Impacto: la respuesta conserva la identidad lógica de la operación, mientras
  `GET` conserva semántica dinámica del snapshot activo. Una función pura
  compartida impide que ambos cálculos diverjan.

### Lectura optimista del snapshot activo

- Decisión: validar revisión y snapshot después de cada lectura pública, con
  tres intentos máximos.
- Alternativas: lock global, transacción Mongo/replica set o devolver el primer
  cálculo aunque el puntero cambie.
- Motivo: el patrón detecta cualquier sustitución sin cambiar despliegue ni
  bloquear importaciones y descarta resultados parciales.
- Impacto: una lectura estable añade una segunda consulta del dataset; bajo
  cambio continuo puede responder `409` después del límite, en vez de devolver
  cobertura incoherente o reintentar indefinidamente.

### Estado enum almacenado

- Decisión: persistir los tres estados.
- Alternativa: derivar contando objetos en cada respuesta.
- Motivo: respuestas de listas no incurren en agregaciones/N+1; el servicio
  actualiza estado y snapshot en el mismo cambio atómico.
- Impacto: futuros consumidores pueden filtrar por estado directamente.

### Basename solo como fallback inequívoco

- Decisión: coincidencia exacta primero, basename único después.
- Alternativa: exigir siempre path exacto o descartar directorios siempre.
- Motivo: combina seguridad con compatibilidad con exports COCO que añaden un
  prefijo de directorio.
- Impacto: datasets con basenames duplicados requieren paths exactos.

### Rechazar cajas fuera de límites

- Decisión: abortar, sin clipping.
- Alternativas: recortar o tolerar.
- Motivo: evita alterar ground truth silenciosamente.
- Impacto: errores son accionables y reproducibles.

### `area` opcional

- Decisión: conservar el valor si existe; calcular `width * height` si falta.
- Alternativa: exigirlo siempre.
- Motivo: COCO bbox puede evaluarse con área de caja y el prompt solo exige
  validar el valor cuando aparece.
- Impacto: futuras métricas tienen siempre área disponible.

### Límite de 10 MiB

- Decisión: `COCO_MAX_UPLOAD_BYTES`, 10 MiB por defecto.
- Alternativas: sin límite o valor fijo.
- Motivo: protección razonable y configurable sin nueva dependencia.
- Impacto: despliegues grandes pueden aumentarlo explícitamente.

### Seguridad solo en endpoints nuevos

- Decisión: proteger cobertura/importación sin reescribir rutas legacy.
- Alternativa: autenticar toda la API de datasets.
- Motivo: lo segundo sería un cambio público material fuera de alcance.
- Impacto: la funcionalidad nueva respeta aislamiento; asegurar rutas legacy
  queda como deuda independiente.

## 9. Pruebas y validación

### Iteración de lectura optimista del `GET`

Todos los comandos Docker se ejecutaron desde `backend/`.

| Fase | Comando exacto | Resultado |
|---|---|---|
| Línea base COCO | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | 38 superados, 0 fallidos, 4.07 s. |
| Regresiones nuevas | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py::test_annotation_coverage_read_without_concurrency_uses_single_attempt tests/test_dataset_coco_annotations.py::test_annotation_coverage_read_retries_when_active_snapshot_changes tests/test_dataset_coco_annotations.py::test_annotation_coverage_read_stops_after_retry_limit` | 3 superados, 0 fallidos, 0.46 s. |
| Ruff focalizado | `docker compose exec backend python -m ruff check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | `All checks passed!`. |
| Formato focalizado inicial | `docker compose exec backend python -m ruff format --check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | Detectó el test por reformatear. |
| Formato aplicado | `docker compose exec backend python -m ruff format tests/test_dataset_coco_annotations.py` | 1 archivo reformateado. |
| COCO focalizada final | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | 41 superados, 0 fallidos, 6.27 s. |
| Datasets/seguridad relacionada | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py tests/test_datasets.py tests/test_auth.py tests/test_users.py` | 69 superados, 0 fallidos, 16.27 s. |
| Backend completo | `docker compose exec backend python -m pytest -ra -W always` | 405 superados, 0 fallidos, 2 omitidos por falta de CUDA, 24.33 s. |
| Ruff completo | `docker compose exec backend python -m ruff check app tests` | `All checks passed!`. |
| Formato completo | `docker compose exec backend python -m ruff format --check app tests` | 110 archivos ya formateados. |
| Compilación Python | `docker compose exec backend python -m compileall -q app tests` | Correcta, sin salida. |
| Ausencia de borrado desigual | `rg -n '\$ne' backend/app/services/dataset_annotation_service.py` | Sin resultados. |
| Diff/estado | `git diff --check`, `git diff --stat`, `git status --short`, `git ls-files --others --exclude-standard \| wc -l` | Sin errores de whitespace; se conservan 12 archivos rastreados modificados y 12 no rastreados, sin entradas nuevas fuera de alcance. |
| HEAD | `git rev-parse HEAD` | `dee4f66e6267bcff747e8b620a04124fbd1ba838`, sin commit ni push. |

El primer comando combinado de inspección/línea base usó rutas relativas
incorrectas desde `backend/` y encontró la restricción de `snap-confine`; no
ejecutó tests. Se repitieron inspección y pytest por separado, con los
resultados anteriores. No se ejecutó frontend porque no hubo cambios allí.

### Iteración de coherencia de respuestas concurrentes

Todos los comandos Docker de esta tabla se ejecutaron desde `backend/`.

| Fase | Comando exacto | Resultado |
|---|---|---|
| Línea base focalizada | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py::test_concurrent_import_cleanup_preserves_snapshot_being_prepared tests/test_dataset_coco_annotations.py::test_import_coco_snapshot_updates_coverage_states_and_canonical_records tests/test_dataset_coco_annotations.py::test_new_dataset_annotation_coverage_is_fully_unreviewed tests/test_dataset_coco_annotations.py::test_partial_annotation_write_failure_cleans_only_prepared_snapshot tests/test_coco_annotation_importer.py` | 25 superados, 0 fallidos, 1.12 s. La prueba concurrente anterior pasó en esta ejecución, aunque su orden posterior a liberar B dependía potencialmente del planificador. |
| Función pura y carrera residual | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py::test_stable_coverage_builder_preserves_states_and_category_distribution tests/test_dataset_coco_annotations.py::test_stable_coverage_builder_handles_empty_dataset tests/test_dataset_coco_annotations.py::test_concurrent_import_responses_remain_bound_to_their_own_snapshots tests/test_dataset_coco_annotations.py::test_import_coco_snapshot_updates_coverage_states_and_canonical_records tests/test_dataset_coco_annotations.py::test_partial_annotation_write_failure_cleans_only_prepared_snapshot` | 5 superados, 0 fallidos, 0.88 s. |
| Focalizada intermedia | Mismo comando focalizado final indicado debajo | 38 superados, 0 fallidos, 5.26 s. |
| Relacionada intermedia | Mismo comando relacionado final indicado debajo | 66 superados, 0 fallidos, 17.19 s. |
| Completa intermedia | `docker compose exec backend python -m pytest -ra -W always` | 402 superados, 0 fallidos, 2 omitidos, 20.50 s. |
| Ambas regresiones de persistencia/concurrencia, estado final | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py::test_concurrent_import_responses_remain_bound_to_their_own_snapshots tests/test_dataset_coco_annotations.py::test_partial_annotation_write_failure_cleans_only_prepared_snapshot` | 2 superados, 0 fallidos, 0.21 s. |
| Importador y servicio focalizados, estado final | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | 38 superados, 0 fallidos, 5.78 s. |
| Datasets y seguridad relacionados, estado final | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py tests/test_datasets.py tests/test_auth.py tests/test_users.py` | 66 superados, 0 fallidos, 15.29 s. |
| Suite backend completa, estado final | `docker compose exec backend python -m pytest -ra -W always` | 402 superados, 0 fallidos, 2 omitidos por falta de CUDA, 22.56 s. |
| Ruff focalizado | `docker compose exec backend python -m ruff check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | `All checks passed!`. |
| Formato focalizado inicial | `docker compose exec backend python -m ruff format --check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | Detectó un archivo de test por reformatear. |
| Formato aplicado | `docker compose exec backend python -m ruff format tests/test_dataset_coco_annotations.py` | 1 archivo reformateado. |
| Formato focalizado final | `docker compose exec backend python -m ruff format --check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | 2 archivos ya formateados. |
| Ruff completo final | `docker compose exec backend python -m ruff check app tests` | `All checks passed!`. |
| Formato completo final | `docker compose exec backend python -m ruff format --check app tests` | 110 archivos ya formateados. |
| Compilación Python | `docker compose exec backend python -m compileall -q app tests` | Correcta, sin salida. |
| Ausencia de borrado desigual | `rg -n '\$ne' backend/app/services/dataset_annotation_service.py` | Sin resultados. |
| Diff rastreado | `git diff --check` | Correcto, sin salida; `git diff --stat` conserva las 12 modificaciones previas (`324+`, `54-`). |
| Estado final | `git status --short` y `git rev-parse HEAD` | Mismos 12 archivos modificados y 12 no rastreados; `HEAD` sigue en `dee4f66e6267bcff747e8b620a04124fbd1ba838`. |

No hay `mypy`, `pyright` ni script de type-check configurado. El primer intento
combinado de localizarlo y ejecutar `compileall` no llegó a validar nada por un
glob inexistente y una restricción de `snap-confine`; las comprobaciones se
repitieron por separado, se confirmó la ausencia de type-checker y
`compileall` pasó. No se ejecutaron frontend/build porque esta iteración no
modificó frontend ni contrato HTTP.

### Iteración anterior de limpieza concurrente

Todos los comandos Docker de esta tabla se ejecutaron desde `backend/`.

| Fase | Comando exacto | Resultado |
|---|---|---|
| Línea base, antes de modificar | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | 34 superados, 0 fallidos, 5.75 s. |
| Regresiones nuevas | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py::test_concurrent_import_cleanup_preserves_snapshot_being_prepared tests/test_dataset_coco_annotations.py::test_partial_annotation_write_failure_cleans_only_prepared_snapshot` | 2 superados, 0 fallidos, 0.21 s. |
| Servicio/importador focalizados | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | 36 superados, 0 fallidos, 5.77 s. |
| Datasets y seguridad relacionados | `docker compose exec backend python -m pytest -ra -W always tests/test_dataset_coco_annotations.py tests/test_datasets.py tests/test_auth.py tests/test_users.py` | 64 superados, 0 fallidos, 14.70 s. |
| Suite backend completa | `docker compose exec backend python -m pytest -ra -W always` | 400 superados, 0 fallidos, 2 omitidos por falta de CUDA, 21.62 s. |
| Ruff focalizado | `docker compose exec backend python -m ruff check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | `All checks passed!`. |
| Formato focalizado | `docker compose exec backend python -m ruff format --check app/services/dataset_annotation_service.py tests/test_dataset_coco_annotations.py` | 2 archivos ya formateados. |
| Ruff completo | `docker compose exec backend python -m ruff check app tests` | `All checks passed!`. |
| Formato completo | `docker compose exec backend python -m ruff format --check app tests` | 110 archivos ya formateados. |
| Ausencia de borrado desigual | `rg -n '\$ne' backend/app/services/dataset_annotation_service.py` | Sin resultados. |
| Integridad del diff rastreado | `git diff --check` | Correcto, sin salida. |
| Estado final | `git status --short` | Se conservan los 12 archivos modificados y 12 archivos no rastreados de la implementación COCO; no apareció ningún archivo fuera de alcance. |

Incidencias operativas no bloqueantes:

- La primera invocación de línea base se lanzó desde la raíz del repositorio y
  Docker respondió `no configuration file provided: not found`; se repitió
  desde `backend/` y pasó.
- Una invocación combinada de dos comandos Ruff encontró una restricción de
  `snap-confine`; ambos comandos se repitieron por separado con el permiso ya
  aprobado y pasaron.
- No se repitieron tests/build/lint del frontend porque esta iteración no
  modificó frontend ni su contrato. Sus resultados de la implementación
  principal se conservan en el histórico siguiente.

### Histórico de la implementación principal

| Fase | Comando exacto | Propósito | Resultado |
|---|---|---|---|
| Estado previo | `docker compose exec backend python -m pytest -ra -W always tests/test_datasets.py tests/test_auth.py tests/test_users.py` | Línea base de datasets/autenticación | 49 superados, 0 fallidos, 9.51 s. |
| Intermedia | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py tests/test_datasets.py` | Primera ejecución COCO | 43 superados, 11 fallidos, 17.19 s. Detectó `await aggregate`, tolerancia float y clientes async ligados a otro loop. |
| Intermedia | mismo comando | Segunda ejecución | 53 superados, 1 fallido, 12.79 s. El test de índices debía recrearlos tras el `drop_database` autouse. |
| Intermedia | mismo comando | Correcciones completas | 54 superados, 0 fallidos, 12.75 s. |
| Focalizada final | `docker compose exec backend python -m pytest -ra -W always tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py tests/test_datasets.py tests/test_auth.py tests/test_users.py` | COCO, API, migración, seguridad y regresión | 83 superados, 0 fallidos, 20.18 s. |
| Arquitectura | `docker compose exec backend python -m pytest -ra -W always tests/test_service_error_architecture.py tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py tests/test_datasets.py` | Errores de servicio y COCO | 57 superados, 0 fallidos, 11.49 s. |
| Suite completa intermedia | `docker compose exec backend python -m pytest -ra -W always` | Regresión global | 397 superados, 2 omitidos, 1 fallido: capturas amplias nuevas prohibidas por arquitectura. |
| Suite completa final | `docker compose exec backend python -m pytest -ra -W always` | Regresión global final | 398 superados, 0 fallidos, 2 omitidos por falta de CUDA, 28.08 s. |
| Migración | incluida en `tests/test_dataset_coco_annotations.py` | Upgrade, downgrade, campos legacy e índices | Superada dentro de suites focalizada/completa. |
| Lint Python | `docker compose exec backend python -m ruff check app tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | Lint backend | `All checks passed!`. |
| Formato Python | `docker compose exec backend python -m ruff format --check app tests/test_coco_annotation_importer.py tests/test_dataset_coco_annotations.py` | Formato backend | 75 archivos ya formateados. |
| Test COCO frontend | `npm run test:coco` | SSR y utilidades de importación | 1 entrada de test superada, 0 fallidas; el archivo define 6 casos, 957 ms. |
| Test frontend existente | `npm run test:gradcam` | No regresión de suite existente | 1 entrada superada, 0 fallidas. |
| Build frontend | `npm run build` | Build de producción | Correcto; 633 módulos, 1.01 s. Advertencia no bloqueante por chunk de 736.61 kB. |
| Lint frontend focal | `npx eslint src/components/CocoAnnotationPanel.jsx src/components/ImageGallery.jsx src/pages/DatasetManager.jsx src/api/client.js src/api/datasets.js src/utils/cocoAnnotations.js tests/cocoAnnotations.test.js` | Todos los JS/JSX nuevos o modificados | Correcto, 0 errores/advertencias. |
| Lint frontend global | `npm run lint` | Suite global | Bloqueo preexistente: 11 errores y 1 advertencia por efectos/contextos en archivos previos; ningún archivo nuevo añade errores. |
| Diff | `git diff --check` | Espacios/conflictos | Correcto, sin salida. |

No hay script de type-check independiente en el frontend ni backend.

## 10. Cobertura de casos

1. Imagen existente `unreviewed`:
   `test_migration_marks_existing_images_unreviewed_and_is_reversible`.
2. Relaciones/restricciones:
   `test_category_and_annotation_unique_indexes_are_enforced` y
   `test_import_coco_snapshot_updates_coverage_states_and_canonical_records`.
3. Cascadas:
   `test_deleting_image_and_dataset_removes_annotation_records`.
4. Precisión:
   `test_coco_importer_creates_canonical_snapshot_with_full_precision` y test
   de persistencia HTTP.
5. Importación con objetos:
   `test_import_coco_snapshot_updates_coverage_states_and_canonical_records`.
6. Presente sin anotaciones:
   el mismo test verifica `reviewed_empty`.
7. Presente con objetos:
   el mismo test verifica `reviewed_with_objects`.
8. Ausente del COCO:
   el mismo test verifica `unreviewed`.
9. Cobertura parcial:
   2/3 = `66.67%` en el mismo test.
10. Reimportación:
    `test_reimport_is_idempotent_and_replacement_removes_old_snapshot`.
11. Reemplazo:
    el mismo test reemplaza por snapshot sin categorías/objetos.
12. IDs/area/iscrowd:
    test de importación y consulta canónica.
13. `xywh -> xyxy`:
    tests de importador y persistencia.
14. JSON malformado:
    `test_coco_importer_rejects_malformed_json` y rollback parametrizado.
15. IDs duplicados:
    `test_coco_importer_rejects_invalid_payloads` para categorías, imágenes y
    anotaciones.
16. Referencia a imagen inexistente:
    caso parametrizado del importador.
17. Referencia a categoría inexistente:
    caso parametrizado y rollback HTTP.
18. Imagen no encontrada:
    `test_coco_importer_rejects_unmatched_dataset_image`.
19. Coincidencia ambigua:
    `test_coco_importer_rejects_ambiguous_basename`.
20. Dimensiones inválidas/incompatibles:
    dos casos parametrizados.
21. Caja degenerada:
    caso parametrizado con ancho cero.
22. Caja no finita:
    caso con `NaN`.
23. Caja fuera de límites:
    importador y rollback HTTP.
24. Ruta maliciosa:
    importador, rollback y upload con filename inseguro.
25. Snapshot intacto:
    `test_invalid_import_preserves_previous_snapshot`, cinco errores
    parametrizados.
26. Cobertura/distribución:
    `test_new_dataset_annotation_coverage_is_fully_unreviewed` y test de
    snapshot parcial.
27. Respuesta coherente:
    test HTTP compara respuesta, estados y documentos canónicos.
28. No autenticado:
    `test_annotation_endpoints_require_owner`.
29. Sin propiedad:
    el mismo test cubre GET y POST con segundo usuario.
30. Dataset inexistente:
    el mismo test verifica `404`.
31. Tipo/tamaño:
    `test_annotation_upload_rejects_type_size_and_unsafe_name`.
32. Renderizado de cobertura:
    `coverage summary renders counts and category distribution`, SSR real con
    Vite y `react-dom/server`.
33. Importación y actualización:
    `successful imports replace the visible coverage with the API result`; el
    componente refresca galería mediante callback.
34. Errores:
    `structured validation errors expose actionable context and rollback`.
35. Confirmación:
    `existing snapshots require replacement confirmation`.
36. Carga/bloqueo:
    `duplicate submissions are blocked while an import is running`.
37. No regresión clasificación:
    `tests/test_datasets.py`, suite completa backend y build frontend.
38. Intercalado concurrente crítico:
    `test_concurrent_import_responses_remain_bound_to_their_own_snapshots`
    fuerza A-activa/B-prepara/A-limpia-el-anterior/B-sobrevive/
    B-activa-y-limpia-A/A-responde. Antes de reanudar A confirma que sus
    documentos ya no existen; verifica respuestas distintas y exactas de A/B,
    cobertura pública de B, IDs de limpieza, revisión y estado final sin
    duplicados ni huérfanos.
39. Escritura parcial:
    `test_partial_annotation_write_failure_cleans_only_prepared_snapshot`
    inyecta el fallo después de categorías y antes de persistir anotaciones;
    verifica rollback propio, revisión/puntero/cobertura anteriores y otra
    preparación intactos.
40. Agregación estable:
    `test_stable_coverage_builder_preserves_states_and_category_distribution`
    cubre objetos, categoría sin objetos, los tres estados, distribución,
    orden y porcentaje;
    `test_stable_coverage_builder_handles_empty_dataset` cubre cero imágenes.
41. Paridad importación/consulta:
    `test_import_coco_snapshot_updates_coverage_states_and_canonical_records`
    exige igualdad completa entre la cobertura estable devuelta por `POST` y
    el `GET` del snapshot activo inmediatamente posterior sin concurrencia.
42. `GET` estable sin concurrencia:
    `test_annotation_coverage_read_without_concurrency_uses_single_attempt`
    verifica dos lecturas del dataset —inicial y validación obligatoria—, una
    consulta de categorías y una agregación de objetos.
43. Sustitución durante el `GET`:
    `test_annotation_coverage_read_retries_when_active_snapshot_changes`
    pausa A durante su agregado, activa B y elimina A; confirma que se descarta
    el primer cálculo y se devuelve exclusivamente B tras el reintento.
44. Límite bajo cambio continuo:
    `test_annotation_coverage_read_stops_after_retry_limit` fuerza una revisión
    distinta en cada validación, verifica exactamente tres rondas y el
    `ConflictError` final sin modificar el dataset persistido.

## 11. Diff y alcance

`git diff --stat` de los archivos ya rastreados sigue mostrando la
implementación principal:

```text
12 files changed, 324 insertions(+), 54 deletions(-)
```

Archivos modificados:

- `backend/app/api/routes/datasets.py`
- `backend/app/core/errors.py`
- `backend/app/db/collections.py`
- `backend/app/main.py`
- `backend/app/schemas/datasets.py`
- `backend/app/services/dataset_service.py`
- `frontend/package.json`
- `frontend/src/api/client.js`
- `frontend/src/api/datasets.js`
- `frontend/src/components/ImageGallery.jsx`
- `frontend/src/index.css`
- `frontend/src/pages/DatasetManager.jsx`

Archivos nuevos:

- `backend/app/db/migrations.py`
- `backend/app/services/annotation_importers/__init__.py`
- `backend/app/services/annotation_importers/base.py`
- `backend/app/services/annotation_importers/coco.py`
- `backend/app/services/annotation_importers/registry.py`
- `backend/app/services/dataset_annotation_service.py`
- `backend/tests/test_coco_annotation_importer.py`
- `backend/tests/test_dataset_coco_annotations.py`
- `frontend/src/components/CocoAnnotationPanel.jsx`
- `frontend/src/utils/cocoAnnotations.js`
- `frontend/tests/cocoAnnotations.test.js`
- `codex-reports/02-feat-datasets-coco-annotations.md`

Como los tres archivos de estas correcciones ya eran entradas no rastreadas de
la implementación principal, Git no puede separar su delta por iteración. El
alcance revisado manualmente en esta última iteración fue:

- `backend/app/services/dataset_annotation_service.py`: añade
  `ANNOTATION_COVERAGE_MAX_READ_ATTEMPTS` y la validación/reintento optimista en
  `get_annotation_coverage`; no modifica `import_coco_annotations`;
- `backend/tests/test_dataset_coco_annotations.py`: añade dobles acotados para
  coordinar/contar lecturas y los tres casos 42–44;
- `codex-reports/02-feat-datasets-coco-annotations.md`: documenta incidencia,
  algoritmo optimista, costes, pruebas, riesgos y diff.

Los otros 21 archivos modificados/no rastreados ya formaban parte de la
implementación principal y se preservaron. El informe 01 y la implementación
de la sub-issue anterior, presentes en `HEAD`, no se modificaron.

No hubo commit ni push. No se añadieron dependencias, métricas de detección,
YOLO TXT, editor gráfico, overlays ni refactorizaciones generales.

## 12. Riesgos y pendientes

- MongoDB standalone no soporta transacciones multi-documento. La atomicidad
  observable se implementa con snapshots identificados y cambio atómico de
  puntero.
- Un apagado abrupto después de preparar y antes de activar puede dejar
  documentos inactivos. Nunca quedan visibles, pero una importación posterior
  no debe barrerlos porque podrían pertenecer a una preparación concurrente.
  Un job futuro de garbage collection necesitará un protocolo seguro para
  distinguir snapshots huérfanos de imports en curso y queda fuera de alcance.
- Si la limpieza best-effort del snapshot sustituido falla, ese snapshot queda
  inactivo y ocupa almacenamiento, pero el puntero nuevo y sus documentos no
  se invalidan.
- En concurrencia, la respuesta de un `POST` puede describir un snapshot que ya
  no sea el activo cuando llega al cliente; es intencionado porque representa
  el resultado estable de esa operación. Para conocer el estado vigente debe
  usarse `GET .../annotations/coverage`, que siempre consulta el puntero activo.
- Si el dataset cambia durante tres rondas consecutivas del `GET`, se devuelve
  `409` en vez de cobertura potencialmente mezclada. El cliente puede
  reintentar; aumentar el límite indiscriminadamente elevaría carga bajo
  contención y no garantizaría progreso.
- El límite de 10 MiB deberá configurarse al alza para COCO muy grandes,
  considerando memoria y tiempo de validación.
- Los datasets con basenames duplicados requieren que COCO preserve la ruta
  exacta; el fallback ambiguo se rechaza intencionadamente.
- Las rutas legacy de datasets siguen sin exigir autenticación. Los endpoints
  nuevos sí están aislados; securizar toda la API es trabajo separado para no
  romper clientes.
- El lint global frontend continúa bloqueado por 11 errores y 1 advertencia
  preexistentes. El lint de todos los archivos de esta tarea pasa.
- Dos tests GPU se omiten por no disponer de CUDA.
- El build conserva la advertencia previa de chunk principal grande.
- Las métricas COCO futuras deberán leer exclusivamente categorías/objetos del
  `active_annotation_snapshot_id`; no deben leer JSON COCO.
- No se almacena segmentación porque esta sub-issue prepara evaluación COCO de
  bounding boxes, no segmentación.

## 13. Checklist de aceptación

- Cumplido — Existe un esquema canónico para categorías, objetos y cajas.
  Evidencia: entidades canónicas, colecciones e índices separados del
  importador.
- Cumplido — Se exponen los tres estados con invariantes verificadas.
  Evidencia: `AnnotationReviewStatus`, activación conjunta y tests de los tres.
- Cumplido — Los datos existentes migran como `unreviewed`.
  Evidencia: test real de upgrade/downgrade.
- Cumplido — COCO JSON puede importarse como snapshot atómico.
  Evidencia: validación, staging, CAS y reemplazo probados.
- Cumplido — Un error de validación no modifica el snapshot anterior.
  Evidencia: rollback HTTP parametrizado.
- Cumplido — La asociación de imágenes es determinista y segura.
  Evidencia: exacta, fallback único, ausente/ambigua/maliciosa probadas.
- Cumplido — Las cajas conservan precisión completa y coordenadas absolutas.
  Evidencia: campos `xyxy` y aserciones sin redondeo.
- Cumplido — El resumen de cobertura es correcto.
  Evidencia: casos 0% y 66.67%, estados y distribución.
- Cumplido — Backend y frontend permiten importar y consultar anotaciones.
  Evidencia: endpoints, API frontend, panel integrado y SSR/build.
- Cumplido — Se respetan propiedad, autenticación y aislamiento.
  Evidencia: no autenticado, segundo usuario y dataset inexistente.
- Cumplido — No se rompe la gestión de clasificación.
  Evidencia: 405 tests backend pasan en esta iteración y el build frontend
  anterior fue correcto; no hubo cambios frontend posteriores.
- Cumplido — Una importación exitosa limpia solo el snapshot anterior que
  observó y sustituyó, y no ejecuta limpieza general cuando ese ID es nulo.
  Evidencia: guarda exacta en `import_coco_annotations` y ausencia de filtro de
  desigualdad en el camino normal.
- Cumplido — Una preparación concurrente no puede ser borrada por el ganador
  anterior.
  Evidencia:
  `test_concurrent_import_responses_remain_bound_to_their_own_snapshots`.
- Cumplido — Un fallo parcial limpia solo el ID propio sin cambiar snapshot
  activo, revisión ni cobertura, y conserva otras preparaciones.
  Evidencia:
  `test_partial_annotation_write_failure_cleans_only_prepared_snapshot`.
- Cumplido — El CAS conserva propietario, revisión y snapshot activo
  observados; un perdedor limpia únicamente su preparación.
  Evidencia: filtro de activación y bloque de rollback existentes, preservados.
- Cumplido — La respuesta de importación no relee documentos que B pueda
  retirar y queda ligada al contenido validado de su propia operación.
  Evidencia: `import_coverage` se construye antes de persistir y la prueba
  concurrente elimina físicamente A antes de permitir que A responda.
- Cumplido — A y B devuelven sus coberturas diferentes y el `GET` posterior
  describe solo B.
  Evidencia:
  `test_concurrent_import_responses_remain_bound_to_their_own_snapshots`.
- Cumplido — Importación y consulta pública comparten todos los campos y reglas
  de agregación sin compartir la fuente temporal.
  Evidencia: `_build_annotation_coverage`, sus dos tests unitarios y la
  aserción de paridad `POST`/`GET` sin concurrencia.
- Cumplido — El `GET` solo devuelve si revisión y snapshot coinciden antes y
  después de consultar categorías/objetos; si no, descarta y reintenta.
  Evidencia: bucle acotado de `get_annotation_coverage` y
  `test_annotation_coverage_read_retries_when_active_snapshot_changes`.
- Cumplido — El camino normal no repite consultas de categorías/objetos.
  Evidencia:
  `test_annotation_coverage_read_without_concurrency_uses_single_attempt`.
- Cumplido — El número de intentos está acotado y el agotamiento conserva el
  manejo de errores existente.
  Evidencia: `ANNOTATION_COVERAGE_MAX_READ_ATTEMPTS = 3` y
  `test_annotation_coverage_read_stops_after_retry_limit`.
- Cumplido — El cálculo estable del `POST`, CAS y limpiezas exactas no se
  modificaron en esta iteración.
  Evidencia: diff manual de los tres archivos en alcance y regresiones previas.
- Cumplido — Revisión del dataset, identidad de snapshot y versión contractual
  están diferenciadas; no se añadió `schema_version` ni cambió la API.
  Evidencia: sección 3 y ausencia de cambios en rutas/schemas en esta iteración.
- Cumplido — No existe lógica de métricas ni dependencia COCOeval.
  Evidencia: búsqueda final y ausencia de cambios en `ml/metrics`/dependencias.
- Cumplido — Las operaciones evitan N+1 y commits por objeto.
  Evidencia: `insert_many`, activación única y consultas fijas de cobertura.
- Cumplido — Migración, tests, lint, tipos y build relevantes pasan o los
  bloqueos quedan demostrados.
  Evidencia: migración probada, Ruff/scoped ESLint/build correctos; deuda global
  y ausencia de type-check script documentadas.
- Cumplido — El diff permanece dentro del alcance.
  Evidencia: 24 archivos, todos de modelo, importación, API, UI, tests o informe.
- Cumplido — Se genera el informe obligatorio.
  Evidencia: `codex-reports/02-feat-datasets-coco-annotations.md`.
