#!/bin/sh
# Descarga e importa datasets de ejemplo ejecutando Python en Docker.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
COMPOSE="$ROOT/compose.yaml"
DATA="$ROOT/example_datasets"

usage() {
  cat <<'EOF'
Uso: sh scripts/datasets/prepare_examples.sh [opciones]

  --count N          10, 20, ..., 100 imágenes por dataset (100 por defecto)
  --task T           both (por defecto), classification o detection
  --prepare-only     Descargar y generar las etiquetas/COCO, sin importar en la app
  --email CORREO     Usuario existente de la plataforma (pide contraseña)
  --help             Mostrar esta ayuda

Por defecto prepara e importa clasificación + detección con etiquetas y COCO.
Para importar: crea una cuenta en http://localhost:5173 si aún no tienes.
El script te pedirá el CORREO y la CONTRASEÑA de esa cuenta (no los de GitHub).
La contraseña se introduce sin mostrarse. No se guarda en los archivos.
No necesitas usar un comando «import»: las imágenes y etiquetas/anotaciones
se suben automáticamente a la plataforma al terminar la preparación.

Ejemplos:
  sh scripts/datasets/prepare_examples.sh
  sh scripts/datasets/prepare_examples.sh --count 20
  sh scripts/datasets/prepare_examples.sh --email usuario@ejemplo.com
  sh scripts/datasets/prepare_examples.sh --prepare-only

Requiere Docker Compose y una cuenta de la aplicación para importar.
EOF
}

COUNT=100
TASK=both
PREPARE_ONLY=0
EMAIL=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --count)
      [ "$#" -ge 2 ] || { echo 'Falta un valor para --count' >&2; exit 2; }
      COUNT=$2; shift 2 ;;
    --task)
      [ "$#" -ge 2 ] || { echo 'Falta un valor para --task' >&2; exit 2; }
      TASK=$2; shift 2 ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --email)
      [ "$#" -ge 2 ] || { echo 'Falta un valor para --email' >&2; exit 2; }
      EMAIL=$2; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Opción desconocida: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$COUNT" in
  10|20|30|40|50|60|70|80|90|100) : ;;
  *) echo '--count debe ser 10, 20, ..., 100' >&2; exit 2 ;;
esac
case "$TASK" in
  both|classification|detection) : ;;
  *) echo '--task debe ser both, classification o detection' >&2; exit 2 ;;
esac

command -v docker >/dev/null 2>&1 || { echo 'Se necesita Docker instalado.' >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo 'Se necesita Docker Compose v2.' >&2; exit 1; }
[ -f "$COMPOSE" ] || { echo 'No se encontró compose.yaml en la raíz.' >&2; exit 1; }
mkdir -p "$DATA"

# La importación necesita que la API esté levantada. Si no existe aún,
# aprovechamos el lanzador general, que crea .env sin sobrescribir secretos.
if [ "$PREPARE_ONLY" -eq 0 ]; then
  if ! docker compose -f "$COMPOSE" ps --status running --services | grep -qx 'backend'; then
    echo 'Iniciando la plataforma para importar los datasets...'
    sh "$ROOT/start.sh"
  fi
fi

# El backend ya incluye requests y Pillow. No se instala Python en el host.
echo 'Preparando el entorno Docker para los datasets...'
docker compose -f "$COMPOSE" build backend

# Evitar archivos propiedad de root en instalaciones Linux. En otros entornos
# Docker Desktop acepta el usuario asignado por su integración de archivos.
USER_OPT=
if command -v id >/dev/null 2>&1; then
  USER_OPT="$(id -u):$(id -g)"
fi

# Los datos de acceso los solicita Python dentro del contenedor mediante
# input() y getpass(). Nunca se pide pasar la contraseña como argumento.
if [ "$PREPARE_ONLY" -eq 0 ]; then
  printf '\n=== Importación automática en la plataforma ===\n'
  printf 'Usa una cuenta creada en http://localhost:5173 (NO la de GitHub).\n'
  if [ -n "$EMAIL" ]; then
    printf 'Correo indicado mediante --email. Se solicitará solo la contraseña.\n'
  else
    printf 'Introduce el correo de tu cuenta cuando aparezca la pregunta.\n'
  fi
  printf 'Introduce después su contraseña: NO se mostrará al escribirla.\n'
  printf 'No debes usar otro comando: se importarán las imágenes, las etiquetas\n'
  printf 'de clasificación y las anotaciones COCO automáticamente.\n\n'
else
  printf '\nModo preparación: no se solicitan credenciales ni se importa nada.\n'
  printf 'Para importarlo más adelante, repite el comando sin --prepare-only.\n\n'
fi

# Se conserva la consola interactiva para solicitar la contraseña sin eco.
# No pasamos contraseñas en argumentos, URLs ni variables de entorno.
set -- python /opt/tfg-datasets/prepare_examples.py --count "$COUNT" --task "$TASK" --data-dir /datasets --api-base http://backend:8000
if [ "$PREPARE_ONLY" -eq 1 ]; then
  set -- "$@" --prepare-only
fi
if [ -n "$EMAIL" ]; then
  set -- "$@" --email "$EMAIL"
fi
if [ -t 0 ]; then
  if [ -n "$USER_OPT" ]; then
    docker compose -f "$COMPOSE" run --rm --no-deps --user "$USER_OPT" \
      -v "$ROOT/scripts/datasets:/opt/tfg-datasets:ro" \
      -v "$DATA:/datasets" backend "$@"
  else
    docker compose -f "$COMPOSE" run --rm --no-deps \
      -v "$ROOT/scripts/datasets:/opt/tfg-datasets:ro" \
      -v "$DATA:/datasets" backend "$@"
  fi
else
  if [ "$PREPARE_ONLY" -eq 0 ]; then
    echo 'La importación requiere una terminal para el login. Usa --prepare-only en modo no interactivo.' >&2
    exit 2
  fi
  docker compose -f "$COMPOSE" run --rm --no-deps -T \
    -v "$ROOT/scripts/datasets:/opt/tfg-datasets:ro" \
    -v "$DATA:/datasets" backend "$@"
fi
