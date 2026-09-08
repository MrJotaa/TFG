# Plataforma de análisis de robustez adversaria

Aplicación web desarrollada como Trabajo de Fin de Grado para analizar la robustez de modelos de visión por computador frente a ataques adversarios.

La plataforma integra tareas de clasificación y detección de objetos, gestión de conjuntos de datos, generación de ataques adversarios, evaluación de modelos, técnicas de explicabilidad e historial persistente de análisis y evaluaciones.

## Arquitectura

La aplicación está compuesta por tres servicios:

* **Frontend:** React + Vite, compilado y servido mediante Nginx.
* **Backend:** FastAPI y Python.
* **Base de datos:** MongoDB.

Docker Compose se utiliza como punto de entrada único para construir y ejecutar todos los servicios.

```text
Navegador
    │
    ├── http://localhost:5173
    │       │
    │       └── Frontend (Nginx + React)
    │
    └── http://localhost:8000
            │
            └── Backend (FastAPI)
                    │
                    └── MongoDB
```

## Arquitectura interna

### Backend

El backend sigue una organización modular:

* `app/api/routes/`: definición de endpoints HTTP y adaptación entre las peticiones de la API y los servicios.
* `app/services/`: lógica de aplicación, evaluación, persistencia e historial.
* `app/ml/`: modelos, ataques adversarios, métricas, adaptadores y técnicas de explicabilidad.
* `app/schemas/`: contratos y modelos de datos utilizados por la API.
* `app/db/`: conexión con MongoDB y acceso a las colecciones.
* `app/core/`: infraestructura transversal, gestión de errores, cancelación y  compatibilidad entre tareas.
* `app/utils/`: utilidades compartidas y gestión de artefactos y ficheros.

FastAPI genera automáticamente la documentación detallada de los endpoints y sus contratos, disponible en `http://localhost:8000/docs`.

### Frontend

El frontend se organiza principalmente en:

* `src/pages/`: vistas principales de la aplicación.
* `src/components/`: componentes reutilizables de interfaz.
* `src/api/`: comunicación con la API del backend.
* `src/context/`: estado y contexto compartido de la aplicación.
* `src/utils/`: lógica auxiliar y transformaciones reutilizables.

Esta separación mantiene desacopladas la interfaz, la comunicación con el backend y la lógica específica de cada funcionalidad.

## Requisitos

Para ejecutar la aplicación únicamente es necesario disponer de:

* Docker Engine o Docker Desktop.
* Docker Compose v2.

No es necesario instalar localmente Python, Node.js ni MongoDB.

Se recomienda disponer de conexión a Internet durante el primer uso de los modelos, ya que algunos pesos preentrenados se descargan bajo demanda.

## Puesta en marcha

Desde la raíz del repositorio:

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

La primera construcción puede tardar más debido a la instalación de las dependencias del backend y frontend.

Una vez iniciados los servicios estarán disponibles:

| Servicio                | Dirección                       |
| ----------------------- | ------------------------------- |
| Aplicación web          | http://localhost:5173           |
| API                     | http://localhost:8000           |
| Documentación Swagger   | http://localhost:8000/docs      |
| Comprobación de MongoDB | http://localhost:8000/health/db |

El estado de los contenedores puede comprobarse mediante:

```bash
docker compose ps
```

MongoDB y el backend deberían aparecer como servicios saludables.

## Configuración

El backend carga su configuración desde:

```text
backend/.env
```

Antes de la primera ejecución debe crearse a partir de la plantilla incluida:

```bash
cp backend/.env.example backend/.env
```

La variable obligatoria es:

```text
JWT_SECRET_KEY
```

Se recomienda generar una clave propia antes de utilizar la aplicación fuera de un entorno local.

Por ejemplo:

```bash
openssl rand -hex 32
```

El backend utiliza por defecto la base de datos:

```text
tfg_adversarial_ai
```

Dentro de Docker Compose, la conexión a MongoDB se configura automáticamente mediante la red interna de Docker, por lo que no es necesario conocer ni configurar la dirección del contenedor.

## Ejecución habitual

Para iniciar la aplicación:

```bash
docker compose up
```

Para reconstruir las imágenes después de realizar cambios:

```bash
docker compose up --build
```

Para ejecutar los servicios en segundo plano:

```bash
docker compose up -d
```

Para detenerlos:

```bash
docker compose down
```

Para consultar el estado:

```bash
docker compose ps
```

Para seguir los logs de todos los servicios:

```bash
docker compose logs -f
```

Para consultar únicamente los logs del backend:

```bash
docker compose logs -f backend
```

Para reiniciar un servicio concreto:

```bash
docker compose restart backend
```

## Persistencia de datos

Los datos de la aplicación no dependen del ciclo de vida de los contenedores.

MongoDB utiliza un volumen persistente de Docker para almacenar, entre otros elementos:

* usuarios;
* conjuntos de datos y sus metadatos;
* historial de análisis;
* historial de evaluaciones;
* configuraciones y resultados persistidos.

Los archivos asociados a datasets, resultados y evaluaciones se almacenan en:

```text
backend/uploads/
```

Esta carpeta se monta dentro del contenedor del backend, por lo que sus datos permanecen disponibles aunque el contenedor sea eliminado o reconstruido.

El comando:

```bash
docker compose down
```

detiene y elimina los contenedores, pero conserva los datos persistentes.

### Retención de resultados

La persistencia física de `backend/uploads/` es independiente de la política de retención gestionada por la aplicación.

Los resultados de análisis y evaluaciones pueden tener una fecha de expiración. Cada usuario puede configurar su periodo de retención y, por defecto, los resultados se conservan durante 24 horas.

Las evaluaciones marcadas como preservadas se mantienen de forma indefinida hasta que el usuario cambie dicha configuración o las elimine explícitamente.

Por tanto, reiniciar o reconstruir los contenedores no elimina los datos, pero la propia aplicación puede eliminar resultados expirados de acuerdo con su política de retención.

### Reinicio completo

El siguiente comando elimina también los volúmenes declarados por Docker Compose:

```bash
docker compose down -v
```

Debe utilizarse únicamente cuando se desee reinicializar deliberadamente la base de datos.

Los archivos existentes en `backend/uploads/` no se eliminan mediante este comando porque se encuentran almacenados directamente en el sistema de archivos del proyecto.

## Dependencias reproducibles

El despliegue utiliza versiones bloqueadas para evitar que una reconstrucción futura instale accidentalmente dependencias diferentes.

### Backend

Las dependencias directas se documentan en:

```text
backend/requirements.txt
```

El entorno exacto utilizado por la imagen Docker se encuentra bloqueado en:

```text
backend/requirements.lock.txt
```

El Dockerfile instala este último fichero durante la construcción.

### Frontend

El frontend utiliza:

```text
frontend/package.json
frontend/package-lock.json
```

La imagen Docker instala las dependencias mediante:

```bash
npm ci
```

por lo que se respeta exactamente el árbol de dependencias registrado en `package-lock.json`.

Las imágenes base de los contenedores y la versión de MongoDB también se encuentran fijadas para mejorar la reproducibilidad del entorno.

## Modelos

Los modelos se cargan de forma diferida cuando son necesarios.

La aplicación incluye modelos para:

* clasificación de imágenes;
* detección de objetos.

Algunos modelos utilizan pesos preentrenados proporcionados por Torchvision, Ultralytics o Hugging Face. La primera utilización de un modelo puede requerir descargar dichos pesos y, por tanto, tardar más que ejecuciones posteriores.

## Pruebas

Las pruebas del backend pueden ejecutarse dentro del contenedor:

```bash
docker compose exec backend pytest
```

También pueden ejecutarse pruebas concretas:

```bash
docker compose exec backend pytest tests/test_health.py -v
```

Para comprobar las dependencias instaladas:

```bash
docker compose exec backend python -m pip check
```

### Rendimiento

El protocolo utilizado para realizar benchmarks reproducibles de las
evaluaciones se encuentra documentado en:

```text
backend/PERFORMANCE.md
```
Este documento describe las métricas registradas, el procedimiento de benchmark y las condiciones necesarias para comparar correctamente distintas versiones.

## Desarrollo del frontend

Si se desea ejecutar el frontend fuera de Docker para trabajar con el servidor de desarrollo de Vite, es necesario disponer de Node.js. Se recomienda utilizar Node.js 22, que es la versión empleada durante la construcción de la imagen Docker.

Desde `frontend/`:

```bash
npm ci
npm run dev
```

Las pruebas pueden ejecutarse mediante:

```bash
npm test
```

El análisis estático puede ejecutarse con:

```bash
npm run lint
```

Y la compilación de producción:

```bash
npm run build
```

Por defecto, el frontend utiliza como backend:

```text
http://127.0.0.1:8000
```

La dirección se encuentra centralizada mediante `VITE_API_BASE_URL`.

## Desarrollo del backend

Para el flujo habitual se recomienda utilizar el contenedor Docker:

```bash
docker compose up backend
```

La API está disponible en:

```text
http://localhost:8000
```

FastAPI expone automáticamente su documentación interactiva en:

```text
http://localhost:8000/docs
```

## Estructura de despliegue

```text
.
├── compose.yaml
├── README.md
│
├── backend/
│   ├── app/
│   ├── tests/
│   ├── uploads/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── .env.example
│   ├── requirements.txt
│   ├── requirements.lock.txt
│   └── pyproject.toml
│
└── frontend/
    ├── src/
    ├── tests/
    ├── public/
    ├── Dockerfile
    ├── .dockerignore
    ├── nginx.conf
    ├── package.json
    └── package-lock.json
```

## Solución de problemas

### El backend no puede conectar con MongoDB

Comprobar el estado de los servicios:

```bash
docker compose ps
```

Y consultar los logs:

```bash
docker compose logs mongo
docker compose logs backend
```

El backend espera a que el `healthcheck` de MongoDB sea satisfactorio antes de arrancar.

### Un puerto ya está ocupado

La configuración utiliza los puertos:

```text
5173  Frontend
8000  Backend
```

Debe comprobarse que no exista otro proceso o contenedor utilizando alguno de ellos.

Por ejemplo:

```bash
docker ps
```

### Reconstrucción completa de las imágenes

Si se desea reconstruir las imágenes ignorando la caché:

```bash
docker compose build --no-cache
docker compose up
```

Esta operación no elimina los datos persistentes.

### Los modelos tardan en responder por primera vez

Los modelos se cargan bajo demanda y algunos pesos preentrenados pueden descargarse durante su primera utilización. Por ello, la primera inferencia puede necesitar más tiempo que las siguientes.
