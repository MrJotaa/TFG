#!/bin/sh
# Stop the TFG platform or uninstall its Docker resources.
# Does not modify local project files. Requires Docker Compose v2.
set -eu

usage() {
    printf 'Uso: sh stop.sh [stop|uninstall]\n'
    printf '  stop (por defecto): detener los servicios sin eliminar contenedores ni datos.\n'
    printf '  uninstall: eliminar servicios, redes, imagenes locales y volumen de MongoDB.\n'
}

if [ "$#" -gt 1 ]; then
    usage >&2
    exit 2
fi
case "${1:-stop}" in
    stop|uninstall) action=${1:-stop} ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

fatal() { printf 'Error: %s\n' "$*" >&2; exit 1; }
log() { printf '%s\n' "$*"; }

[ -f compose.yaml ] || fatal 'No se encuentra compose.yaml junto a stop.sh.'
command -v docker >/dev/null 2>&1 || fatal 'Docker no esta instalado o no esta en PATH.'
docker compose version >/dev/null 2>&1 || fatal 'Se necesita Docker Compose v2 (docker compose).'
docker info >/dev/null 2>&1 || fatal 'Docker no responde. Inicia Docker Engine/Desktop y revisa los permisos.'

if [ "$action" = stop ]; then
    log 'Deteniendo los servicios de la plataforma (se conservan contenedores y datos)...'
    docker compose -f compose.yaml stop
    log 'Servicios detenidos. Se conservan el volumen de MongoDB y los archivos locales.'
    exit 0
fi

log 'ATENCION: se eliminaran los contenedores y redes de la plataforma,'
log 'las imagenes locales de Compose y el volumen persistente de MongoDB.'
log 'Se perderan las cuentas, el historial y los metadatos guardados en MongoDB.'
log 'No se borraran backend/uploads/, example_datasets/, backend/.env ni el codigo fuente.'
printf 'Escribe DESINSTALAR para confirmar: '
if ! IFS= read -r confirmation; then
    fatal 'No se recibio confirmacion; no se ha eliminado nada.'
fi
if [ "$confirmation" != 'DESINSTALAR' ]; then
    log 'Desinstalacion cancelada. No se ha eliminado nada.'
    exit 0
fi

docker compose -f compose.yaml down --volumes --rmi local --remove-orphans
log 'Recursos de Docker retirados. Los archivos locales se han conservado.'
