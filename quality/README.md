# Análisis de calidad del software

Este directorio documenta la estrategia de análisis estático, pruebas, cobertura y control de calidad empleada en el TFG.

## Herramientas

El proyecto utiliza las siguientes herramientas:

- **SonarQube Community Build** para análisis estático, métricas y Quality Gate.
- **SonarScanner CLI** para enviar los análisis a SonarQube.
- **Ruff** para análisis estático del backend.
- **Pytest** y **pytest-cov** para pruebas y cobertura del backend.
- **ESLint** para análisis estático del frontend.
- **Node Test Runner** para las pruebas del frontend.
- **c8** para generar la cobertura del frontend en formato LCOV.

Se mantienen dos proyectos independientes en SonarQube:

- `tfg-backend`
- `tfg-frontend`

## Quality Gate

El proyecto aplica un Quality Gate orientado principalmente al código nuevo.

| Métrica | Requisito |
|---|---:|
| Nuevas incidencias | 0 |
| Security Hotspots sin revisar | 0 |
| Cobertura en código nuevo | >= 80 % |
| Duplicación en código nuevo | <= 3 % |

La definición de **New Code** debe configurarse en SonarQube mediante una referencia estable, preferentemente **Previous version**.

La versión analizada se declara en los archivos `sonar-project.properties` mediante:

```properties
sonar.projectVersion=1.0.0
```

Al incrementar la versión se establece una nueva referencia para distinguir el código existente del código introducido posteriormente.

## Requisitos

Para ejecutar el análisis completo se necesita:

- Docker.
- Docker Compose.
- Node.js y npm.
- SonarQube Community Build en ejecución.
- El backend Docker iniciado.
- Dos tokens de SonarQube, uno para cada proyecto.

Por defecto se asume SonarQube disponible en:

```text
http://localhost:9000
```

## Tokens de SonarQube

El backend y el frontend utilizan proyectos diferentes en SonarQube y, por tanto, pueden utilizar tokens diferentes.

Los tokens **no deben almacenarse en el repositorio**.

Antes de ejecutar el análisis se exportan como variables de entorno:

```bash
export SONAR_BACKEND_TOKEN="<token-backend>"
export SONAR_FRONTEND_TOKEN="<token-frontend>"
```

Opcionalmente puede cambiarse la URL de SonarQube:

```bash
export SONAR_HOST_URL="http://localhost:9000"
```

## Ejecución automatizada

El análisis completo debe poder ejecutarse desde la raíz del repositorio mediante:

```bash
./scripts/quality.sh
```

El script debe ejecutar secuencialmente:

1. Ruff sobre el backend.
2. Tests del backend.
3. Generación de cobertura del backend.
4. ESLint sobre el frontend.
5. Tests del frontend.
6. Build de producción del frontend.
7. Generación de cobertura del frontend.
8. Análisis SonarQube del backend utilizando `SONAR_BACKEND_TOKEN`.
9. Análisis SonarQube del frontend utilizando `SONAR_FRONTEND_TOKEN`.
10. Espera y validación de ambos Quality Gates.

El proceso debe finalizar con código distinto de cero si cualquiera de las validaciones falla.

## Cobertura del backend

La cobertura se genera con Pytest y pytest-cov:

```bash
docker compose exec -T backend \
  pytest \
  --cov=app \
  --cov-report=term-missing \
  --cov-report=xml:coverage/coverage.xml
```

El informe utilizado por SonarQube es:

```text
backend/coverage/coverage.xml
```

## Cobertura del frontend

La cobertura se genera mediante c8:

```bash
npm run test:coverage
```

El informe utilizado por SonarQube es:

```text
frontend/coverage/lcov.info
```

La cobertura mostrada por c8 y la cobertura global mostrada por SonarQube no representan necesariamente el mismo conjunto de archivos. c8 mide los módulos cargados por la suite de pruebas, mientras que SonarQube puede incluir además componentes y páginas React que forman parte del código fuente analizado aunque no hayan sido cargados durante los tests.

## Ejecución manual de SonarScanner

### Backend

Desde la raíz del repositorio:

```bash
docker run --rm \
  --network host \
  -e SONAR_HOST_URL="${SONAR_HOST_URL:-http://localhost:9000}" \
  -e SONAR_TOKEN="$SONAR_BACKEND_TOKEN" \
  -v "$PWD:/usr/src" \
  -w /usr/src/backend \
  sonarsource/sonar-scanner-cli:12.1.0.3233_8.0.1 \
  -Dsonar.qualitygate.wait=true \
  -Dsonar.qualitygate.timeout=300
```

### Frontend

Desde la raíz del repositorio:

```bash
docker run --rm \
  --network host \
  -e SONAR_HOST_URL="${SONAR_HOST_URL:-http://localhost:9000}" \
  -e SONAR_TOKEN="$SONAR_FRONTEND_TOKEN" \
  -v "$PWD:/usr/src" \
  -w /usr/src/frontend \
  sonarsource/sonar-scanner-cli:12.1.0.3233_8.0.1 \
  -Dsonar.qualitygate.wait=true \
  -Dsonar.qualitygate.timeout=300
```

## Flujo de trabajo recomendado

Antes de cerrar una versión:

```bash
export SONAR_BACKEND_TOKEN="<token-backend>"
export SONAR_FRONTEND_TOKEN="<token-frontend>"
./scripts/quality.sh
```

La versión solo debe considerarse validada cuando:

- Ruff no detecta errores.
- ESLint no detecta errores.
- Todas las pruebas pasan.
- El frontend compila correctamente.
- Los informes de cobertura se generan correctamente.
- Los dos análisis de SonarQube terminan correctamente.
- Ambos Quality Gates pasan.

## Automatización con GitHub Actions

Si SonarQube permanece accesible únicamente mediante `localhost`, un runner estándar alojado por GitHub no podrá acceder al servidor.

Para automatizar el análisis en cada `push` sin exponer SonarQube a Internet se puede instalar un **GitHub Actions self-hosted runner** en la misma máquina Linux que ejecuta SonarQube.

Los tokens deben configurarse como secretos del repositorio, por ejemplo:

```text
SONAR_BACKEND_TOKEN
SONAR_FRONTEND_TOKEN
```

El checkout del repositorio debe utilizar historial completo (`fetch-depth: 0`) para conservar correctamente la información SCM utilizada por SonarQube.

## Solución de problemas

### `Missing blame information`

Comprobar que:

- los cambios relevantes están registrados en Git;
- el repositorio dispone del historial completo;
- no se está utilizando un clon superficial.

### Cobertura al 0 %

Regenerar los informes de cobertura antes de lanzar SonarScanner.

Backend:

```bash
docker compose exec backend pytest \
  --cov=app \
  --cov-report=xml:coverage/coverage.xml
```

Frontend:

```bash
npm run test:coverage
```

### SonarQube no está accesible

Comprobar que la interfaz responde en:

```text
http://localhost:9000
```

También debe verificarse que los contenedores de SonarQube y PostgreSQL estén iniciados.

### Token inválido

Generar un nuevo token para el proyecto correspondiente y volver a exportarlo:

```bash
export SONAR_BACKEND_TOKEN="<token-backend>"
export SONAR_FRONTEND_TOKEN="<token-frontend>"
```

## Resultados finales de la auditoría

Tras la auditoría y refactorización realizada durante el sprint de calidad, tanto backend como frontend quedaron sin incidencias abiertas de Security, Reliability o Maintainability en SonarQube.

### Backend

- Security: A.
- Reliability: A.
- Maintainability: A.
- Incidencias abiertas: 0.
- Cobertura global superior al 80 %.
- Duplicación inferior al 3 %.

### Frontend

- Security: A.
- Reliability: A.
- Maintainability: A.
- Incidencias abiertas: 0.
- Duplicación global inferior al 3 %.
- La suite instrumentada mediante c8 presenta una cobertura de líneas superior al 90 % sobre los módulos ejercitados por las pruebas.

La cobertura global mostrada por SonarQube es inferior a la de c8 debido a que SonarQube incluye en su universo de análisis componentes y páginas React que no son cargados por la suite actual basada en Node Test Runner.
