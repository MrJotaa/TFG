#!/bin/sh
# Start the TFG platform with automatic, CPU-only, or NVIDIA GPU deployment.
# Requires only Docker Compose v2 and standard POSIX shell utilities.
set -eu

usage() {
    printf 'Uso: sh start.sh [auto|cpu|gpu]\n'
    printf '  auto (por defecto): CUDA si Docker puede usar NVIDIA; CPU si no.\n'
    printf '  cpu: iniciar siempre sin reservar GPU.\n'
    printf '  gpu: exigir GPU NVIDIA y CUDA disponible en PyTorch.\n'
}

case "${1:-auto}" in
    auto|cpu|gpu) mode=${1:-auto} ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then usage >&2; exit 2; fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

fatal() { printf 'Error: %s\n' "$*" >&2; exit 1; }
log() { printf '%s\n' "$*"; }

command -v docker >/dev/null 2>&1 || fatal 'Docker no esta instalado o no esta en PATH.'
docker compose version >/dev/null 2>&1 || fatal 'Se necesita Docker Compose v2 (docker compose).'
docker info >/dev/null 2>&1 || fatal 'Docker no responde. Inicia Docker Engine/Desktop y revisa los permisos.'

make_secret() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    elif command -v od >/dev/null 2>&1 && command -v tr >/dev/null 2>&1 && [ -r /dev/urandom ]; then
        od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
    else
        fatal 'No se puede generar una clave segura (se necesita openssl u od y /dev/urandom).'
    fi
}

ensure_env() {
    env_file=backend/.env
    if [ -f "$env_file" ]; then
        if ! grep -Eq '^[[:space:]]*JWT_SECRET_KEY[[:space:]]*=[[:space:]]*[^[:space:]#]+' "$env_file"; then
            fatal "$env_file existe sin una JWT_SECRET_KEY valida. Corrigelo manualmente; no se ha modificado."
        fi
        return
    fi
    umask 077
    if [ -f backend/.env.example ]; then
        cp backend/.env.example "$env_file"
    else
        : > "$env_file"
    fi
    secret=$(make_secret)
    if grep -Eq '^[[:space:]]*JWT_SECRET_KEY[[:space:]]*=' "$env_file"; then
        # POSIX sed, with temporary output to also work on macOS (no sed -i).
        sed "s/^[[:space:]]*JWT_SECRET_KEY[[:space:]]*=.*/JWT_SECRET_KEY=$secret/" "$env_file" > "$env_file.tmp"
        mv "$env_file.tmp" "$env_file"
    else
        printf '\nJWT_SECRET_KEY=%s\n' "$secret" >> "$env_file"
    fi
    chmod 600 "$env_file" 2>/dev/null || true
    log 'Configuracion privada creada: backend/.env (clave JWT generada).'
}

probe_gpu() {
    docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi -L >/dev/null 2>&1
}

# Runtime probe is useful only on hosts with a visible NVIDIA device.
# Explicit 'gpu' always probes (also covers remote Docker engines).
host_gpu_hint() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        return 0
    fi
    [ -e /dev/nvidiactl ] || [ -d /proc/driver/nvidia/gpus ]
}

ensure_env
# Do not silently risk committing the generated JWT secret into Git.
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if ! git check-ignore -q -- backend/.env; then
        log 'AVISO: backend/.env no figura como ignorado por Git. Añadelo a .gitignore antes de hacer commits.'
    fi
fi
use_gpu=false
if [ "$mode" = gpu ] || { [ "$mode" = auto ] && host_gpu_hint; }; then
    log 'Comprobando acceso de Docker a NVIDIA (puede descargar una imagen de prueba la primera vez)...'
    if probe_gpu; then
        use_gpu=true
    elif [ "$mode" = gpu ]; then
        fatal 'Docker no puede acceder a una GPU NVIDIA. Revisa el controlador, NVIDIA Container Toolkit o Docker Desktop.'
    else
        log 'GPU NVIDIA no utilizable desde Docker: se iniciara en CPU.'
    fi
elif [ "$mode" = auto ]; then
    log 'No se detecta GPU NVIDIA en el anfitrion: se iniciara en CPU (puedes forzar la prueba con gpu).'
fi

if [ "$use_gpu" = true ]; then
    log 'Iniciando servicios con acceso a GPU NVIDIA...'
    docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
    if docker compose -f compose.yaml -f compose.gpu.yaml exec -T backend python -c 'import torch; assert torch.cuda.is_available(), "CUDA no esta disponible en PyTorch"; print("CUDA disponible:", torch.cuda.get_device_name(0))'; then
        log 'Modo GPU confirmado dentro del backend.'
    elif [ "$mode" = gpu ]; then
        fatal 'El contenedor se inicio, pero PyTorch no detecta CUDA. Revisa el controlador y la compatibilidad con CUDA 13.'
    else
        log 'PyTorch no detecta CUDA; recreando el backend sin reserva de GPU.'
        docker compose -f compose.yaml up -d --force-recreate backend frontend
        log 'Modo CPU seleccionado.'
    fi
else
    log 'Iniciando servicios en modo CPU...'
    docker compose -f compose.yaml up -d --build
fi

log 'Aplicacion: http://localhost:5173'
log 'API: http://localhost:8000  |  Estado: docker compose ps'
