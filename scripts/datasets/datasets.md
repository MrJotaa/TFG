# Datasets de ejemplo

Estos conjuntos permiten probar la evaluación supervisada de **clasificación** y **detección**. No contienen imágenes distribuidas con el repositorio: los scripts las obtienen de las fuentes originales.

## Preparación e importación automática

1. Instala Docker y arranca la plataforma con `sh start.sh` (o el lanzador correspondiente a tu sistema).
2. Abre <http://localhost:5173> y **crea una cuenta** (o utiliza una existente).
3. Desde la raíz del repositorio, en una terminal compatible con `sh`, ejecuta:

```sh
sh scripts/datasets/prepare_examples.sh
```

El comando prepara e **importa automáticamente dos datasets de 100 imágenes**: Imagenette (clasificación, 10 clases con etiqueta por imagen) y COCO 2017 (detección, con anotaciones de objetos). Al solicitarlo, introduce el correo y la contraseña de la cuenta de la aplicación. La contraseña no aparece en pantalla y **no se guarda** en los archivos descargados.

No es necesario instalar Python, `requests` ni herramientas de descarga en el equipo: se utiliza la imagen Docker existente del backend. Se necesita conexión a Internet y espacio suficiente para las fuentes oficiales y las imágenes seleccionadas.

### Opciones

```sh
# Ambos datasets, 20 imágenes cada uno (2 por clase en Imagenette)
sh scripts/datasets/prepare_examples.sh --count 20

# Solo clasificación, 50 imágenes
sh scripts/datasets/prepare_examples.sh --task classification --count 50

# Solo detección, 30 imágenes
sh scripts/datasets/prepare_examples.sh --task detection --count 30

# Preparar imágenes, CSV y JSON COCO, pero NO importar a la aplicación
sh scripts/datasets/prepare_examples.sh --prepare-only

# Pasar el correo para evitar escribirlo dos veces; se solicitará la contraseña
sh scripts/datasets/prepare_examples.sh --email cuenta@ejemplo.com
```

El tamaño admitido es **10, 20, …, 100 imágenes por dataset**. Si no se indica nada se preparan 100 para cada tarea. En Imagenette el conjunto está balanceado entre las diez clases; en COCO se seleccionan imágenes anotadas favoreciendo la variedad de categorías (no necesariamente el mismo número por clase).

### Descargas y reintentos

Las descargas se realizan en `example_datasets/` (excluida del control de versiones), con caché de los archivos oficiales y descargas parciales reanudables. Los archivos de imagen se verifican antes de reutilizarse; los reintentos usan esperas progresivas. Si hay un corte prolongado o se interrumpe el comando, **puede ejecutarse de nuevo con las mismas opciones**: conservará los archivos ya preparados, consultará la API y completará únicamente las imágenes o etiquetas pendientes. Los reintentos son finitos: un fallo persistente se notifica claramente y se puede reanudar después.

Se evita repetir la importación de anotaciones si ya existe una cobertura válida. No borres manualmente `import_state.json` durante una importación. Si has eliminado el dataset desde la interfaz, el siguiente intento podrá crearlo de nuevo.

### Archivos generados

```text
example_datasets/
  .cache/                              # archivos descargados de origen
  classification/imagenette_100/
    images/                            # 100 imágenes
    labels.csv                         # etiquetas por nombre de archivo
    import_state.json                  # si se realizó la importación
  detection/coco_100/
    images/                            # 100 imágenes
    instances_subset.json              # anotaciones COCO
    import_state.json                  # si se realizó la importación
```

Los nombres de las carpetas cambian según `--count`. Si se usa `--prepare-only`, los archivos locales están listos para importarse posteriormente ejecutando de nuevo el comando sin esa opción. En la interfaz aparecerán como `TFG Ejemplo Imagenette N (seed 42)` y `TFG Ejemplo COCO N (seed 2026)`.

## Fuentes y reproducibilidad

- **Imagenette:** subconjunto de las 10 clases de ImageNet distribuido por fastai. Se usa `imagenette2-320.tgz`, split de validación, selección balanceada con semilla `42`. Las etiquetas utilizan los nombres canónicos compatibles con ResNet18 de Torchvision.
- **COCO 2017:** anotaciones `instances_val2017.json` del archivo oficial `annotations_trainval2017.zip` e imágenes del conjunto `val2017`; selección por diversidad de categorías con semilla `2026`. Se conserva el catálogo completo de categorías y el JSON COCO correspondiente exclusivamente a las imágenes seleccionadas.

Los datos se descargan de servidores de los proyectos originales. **El repositorio no concede derechos de redistribución sobre las fotografías de terceros**. Consulta las condiciones de las fuentes y las licencias originales antes de redistribuir imágenes. Este mecanismo automatiza su obtención para experimentación local, sin incluirlas en Git ni en los archivos de la release.

## Problemas habituales

- **No hay cuenta:** crea una desde la interfaz antes de ejecutar la importación.
- **Docker no está activo:** inicia Docker Desktop o el servicio Docker y repite el comando.
- **Sin terminal interactiva:** utiliza `--prepare-only` para descargar; la importación de una cuenta requiere introducir sus credenciales por terminal.
- **Corte de conexión:** ejecuta de nuevo el mismo comando. Se reutilizará el progreso existente.
- **No aparece un dataset:** revisa la cuenta con la que iniciaste sesión; los datasets pertenecen al usuario autenticado.
- **Poco espacio:** las fuentes oficiales de COCO incluyen anotaciones de entrenamiento y validación, por lo que el archivo de descarga es considerablemente mayor que el subconjunto de 100 imágenes final.
