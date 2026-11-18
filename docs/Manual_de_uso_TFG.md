# Manual de uso de la plataforma

Este manual describe la puesta en marcha y el uso básico de la plataforma de análisis de robustez adversaria. La aplicación permite trabajar con modelos de **clasificación de imágenes** y **detección de objetos**, aplicar ataques a una imagen, evaluar conjuntos de datos completos y comparar resultados.

## 1. Instalación y puesta en marcha

Se necesitan **Docker Engine o Docker Desktop** y **Docker Compose v2**. No es necesario instalar Python, Node.js ni MongoDB en el equipo. Desde la raíz del repositorio, ejecute el lanzador correspondiente a su sistema operativo.

En Linux, macOS o Git Bash (Windows):

```bash
sh start.sh
```

En Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

De forma predeterminada, los lanzadores comprueban si Docker puede utilizar una GPU NVIDIA y si PyTorch reconoce CUDA. Si no hay una GPU compatible o accesible, el modo automático utiliza la **CPU**. Para seleccionar el dispositivo explícitamente, añada `cpu` o `gpu` al comando; por ejemplo, `sh start.sh cpu` o `sh start.sh gpu`. En PowerShell también puede añadir `cpu` o `gpu` después de `start.ps1`. El modo `gpu` requiere soporte NVIDIA correctamente configurado en Docker y falla si no se puede utilizar CUDA, en lugar de continuar silenciosamente en CPU.

En el primer arranque, los lanzadores crean `backend/.env` con una clave `JWT_SECRET_KEY` aleatoria y segura si todavía no existe. **Nunca sobrescriben un `.env` existente**. La ejecución con GPU requiere controladores NVIDIA compatibles y acceso a la GPU desde Docker; los lanzadores no instalan estos componentes.

Como alternativa, puede iniciar la plataforma directamente con Docker Compose. Antes, asegúrese de disponer de `backend/.env` con una `JWT_SECRET_KEY` configurada; si existe `backend/.env.example`, puede utilizarla como plantilla. El siguiente comando utiliza la configuración estándar, compatible con CPU:

```bash
docker compose up --build
```

Cuando terminen de iniciarse los servicios, abra **http://localhost:5173** en el navegador. La documentación interactiva de la API está en `http://localhost:8000/docs`.

Para comprobar los contenedores utilice `docker compose ps`; para detener la plataforma, `docker compose down`. Este último comando **no borra** la base de datos ni los archivos persistentes. La primera construcción y la primera carga de algunos modelos pueden tardar más, ya que se instalan dependencias y pueden descargarse pesos preentrenados; conviene disponer de conexión a Internet.

## 2. Acceso y preparación de datos

Una vez iniciada la aplicación, abra **http://localhost:5173**. En la pantalla inicial, pulse **Crear cuenta**, introduzca nombre, correo electrónico y contraseña (con confirmación), o utilice **Iniciar sesión** si ya dispone de una cuenta. Tras acceder se muestra el panel principal, **Datos del experimento**.

### Preparación automática de datasets de ejemplo

Con la aplicación iniciada y una cuenta creada, ejecute desde la **raíz del repositorio**:

```bash
# Linux, macOS o Git Bash (Windows)
sh scripts/datasets/prepare_examples.sh
```

En **Windows PowerShell**:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1
```

Ambos lanzadores ejecutan el mismo programa Python dentro de Docker.
El script utiliza Docker para descargar e importar automáticamente **100 imágenes de Imagenette para clasificación**, con sus etiquetas, y **100 imágenes de COCO 2017 para detección**, con sus anotaciones. Solicita el **correo electrónico y la contraseña de la cuenta de la aplicación** (no los de GitHub); la contraseña no se muestra al escribirla. No es necesario instalar Python localmente ni etiquetar manualmente estos conjuntos.

Puede ajustar la cantidad a un múltiplo de 10 entre 10 y 100 imágenes por dataset, o preparar solo una tarea:

```bash
sh scripts/datasets/prepare_examples.sh --count 20
sh scripts/datasets/prepare_examples.sh --task classification --count 50
sh scripts/datasets/prepare_examples.sh --task detection --count 30
```

En PowerShell se emplean parámetros equivalentes, por ejemplo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1 -Count 20
powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1 -Task detection -Count 30
```

Para descargar y preparar los archivos sin importarlos, utilice `--prepare-only` en `sh` o `-PrepareOnly` en PowerShell. Los datos se guardan en `example_datasets/`, fuera del seguimiento de Git. El proceso conserva los archivos preparados y admite reintentos; si se interrumpe, ejecute de nuevo el mismo comando para continuar. La importación requiere una terminal interactiva y conexión con la aplicación. Encontrará los conjuntos ya importados en **Gestionar datasets**, asociados a la cuenta utilizada.

### Preparación manual de datasets propios

También puede pulsar **Gestionar datasets** y después **Nuevo**. Indique el nombre, elija el tipo **Clasificación** o **Detección**, seleccione una o varias imágenes y confirme con **Crear con N imágenes**. En esta pantalla puede seleccionar un conjunto existente, **Añadir imágenes**, cambiar su nombre, inspeccionar imágenes o eliminarlas.

- **Clasificación:** en la pestaña **Imágenes**, seleccione cada imagen y complete **Etiqueta de clase** mediante **Guardar etiqueta**. Las etiquetas deben corresponder a clases reconocibles por el modelo para obtener métricas supervisadas.
- **Detección:** abra **Anotaciones** e **Importar anotaciones** para cargar un archivo JSON en formato COCO con las categorías y cajas de los objetos. Revise el indicador de cobertura; una nueva importación **reemplaza** las anotaciones COCO anteriores del conjunto.

Las etiquetas y anotaciones no son imprescindibles para obtener predicciones, pero sí condicionan la disponibilidad de las métricas supervisadas. Utilice un dataset de la misma tarea que el modelo que vaya a evaluar.

## 3. Configuración y análisis de una imagen

Regrese al panel principal pulsando el nombre **Analizador adversario** en la cabecera. En la columna izquierda, seleccione un **Modelo** y pulse **Activar modelo**; después elija un **Ataque** compatible y, si procede, ajuste sus **Parámetros**. En **Datos**, seleccione un conjunto y una imagen de su galería. Pulse **Analizar imagen**.

La vista de análisis presenta las pestañas **Predicción** (o **Detección** si se utiliza un detector), **Comparación** y **Explicabilidad**. La primera ejecuta el modelo sin perturbación; la segunda aplica el ataque y confronta los resultados originales y adversarios, y la tercera proporciona representaciones visuales. En clasificación se utiliza **Grad-CAM** y, cuando el detector lo permite, en detección se utiliza **Eigen-CAM**. El explorador adversario permite examinar las imágenes y sus diferencias; en detección se muestran también las cajas y las confianzas de los objetos detectados.

Tras aplicar un ataque, pulse **Guardar análisis** si desea incorporarlo de forma permanente al historial: el resultado individual es **temporal** hasta que se guarda. Desde **Comparar este análisis** también puede preparar una segunda ejecución o utilizar otra ya conservada.

## 4. Evaluación de un dataset completo

En el panel principal seleccione un modelo activo, un ataque compatible y el conjunto de datos; **no hace falta seleccionar una imagen**. Pulse **Evaluar dataset** y, en la nueva vista, **Ejecutar evaluación**. En detección puede ajustar además el umbral de **Confianza**, el **IoU de matching** y la opción **Incluir detalle por objeto**. Puede seguir el progreso y cancelar una evaluación en curso.

Al terminar, consulte las pestañas de resultados. En **clasificación** aparecen **Resumen**, **Por clase**, **Matrices**, **Curvas** e **Imágenes**; en **detección**, **Resumen**, **Métricas COCO**, **Por clase**, **Curvas PR**, **Por objeto**, **Imágenes** y **TIDE** cuando corresponda. Se muestran indicadores originales y tras el ataque; las métricas supervisadas solo estarán disponibles si las etiquetas o anotaciones son evaluables. Si ya existe una evaluación equivalente, la plataforma ofrece **Usar resultado existente** o **Ejecutar de nuevo**.

## 5. Comparaciones, historial y perfil

Abra la pestaña **Historial** del panel principal y elija **Evaluaciones** o **Análisis individuales**. Use **Abrir** para recuperar un resultado o **Comparar** para elegirlo como **A** y seleccionar un resultado compatible **B**. Las comparaciones controladas mantienen fijo el dataset y su revisión —y también la imagen en los análisis individuales—, y permiten variar **una sola** dimensión: modelo, ataque o parámetros del ataque.

Las **evaluaciones se registran automáticamente** y, por defecto, caducan a las **24 horas**; desde el menú de acciones del historial puede elegir **Conservar indefinidamente**. Los análisis individuales solo se conservan si se pulsa **Guardar análisis**. Desde el acceso al usuario de la cabecera se abre **Perfil**, donde puede modificar sus datos y contraseña, consultar o eliminar resultados y cambiar la **retención de nuevas evaluaciones** (sin efecto retroactivo). Utilice **Cerrar sesión** al terminar.
