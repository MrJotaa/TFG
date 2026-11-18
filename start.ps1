# PowerShell 5.1+ / PowerShell 7+. Usage: .\start.ps1 [auto|cpu|gpu]
param(
    [ValidateSet('auto', 'cpu', 'gpu')]
    [string]$Mode = 'auto'
)

$ErrorActionPreference = 'Stop'

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) {
        throw "Docker ha devuelto el codigo ${LASTEXITCODE}: docker $($args -join ' ')"
    }
}

function Test-GpuInDocker {
    try {
        & docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi -L *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function New-Secret {
    $bytes = New-Object 'System.Byte[]' 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    return [System.BitConverter]::ToString($bytes).Replace('-', '').ToLowerInvariant()
}

Push-Location $PSScriptRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker no esta instalado o no esta en PATH.'
    }
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Se necesita Docker Compose v2.' }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker no responde. Inicia Docker Desktop/Engine.' }

    $envPath = Join-Path $PSScriptRoot 'backend/.env'
    $template = Join-Path $PSScriptRoot 'backend/.env.example'
    if (Test-Path $envPath) {
        $existing = Get-Content -Path $envPath -Raw
        if ($existing -notmatch '(?m)^\s*JWT_SECRET_KEY\s*=\s*[^\s#]+') {
            throw 'backend/.env existe sin JWT_SECRET_KEY valida. Corrigelo manualmente; no se ha modificado.'
        }
    } else {
        $content = if (Test-Path $template) { Get-Content -Path $template -Raw } else { '' }
        $secret = New-Secret
        if ($content -match '(?m)^\s*JWT_SECRET_KEY\s*=') {
            $content = [regex]::Replace($content, '(?m)^\s*JWT_SECRET_KEY\s*=.*$', "JWT_SECRET_KEY=$secret")
        } else {
            $content = $content.TrimEnd() + "`nJWT_SECRET_KEY=$secret`n"
        }
        # UTF-8 without BOM, compatible with Compose on Windows PowerShell 5.1.
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($envPath, $content, $utf8)
        Write-Host 'Configuracion privada creada: backend/.env (clave JWT generada).'
    }

    if (Get-Command git -ErrorAction SilentlyContinue) {
        & git rev-parse --is-inside-work-tree *> $null
        if ($LASTEXITCODE -eq 0) {
            & git check-ignore -q -- backend/.env *> $null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning 'backend/.env no figura como ignorado por Git. Añadelo a .gitignore antes de hacer commits.'
            }
        }
    }

    $useGpu = $false
    $hostGpu = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            & nvidia-smi -L *> $null
            $hostGpu = $LASTEXITCODE -eq 0
        } catch {
            $hostGpu = $false
        }
    }
    if ($Mode -eq 'gpu' -or ($Mode -eq 'auto' -and $hostGpu)) {
        Write-Host 'Comprobando acceso de Docker a NVIDIA (puede descargar una imagen de prueba)...'
        if (Test-GpuInDocker) {
            $useGpu = $true
        } elseif ($Mode -eq 'gpu') {
            throw 'Docker no puede acceder a NVIDIA. Revisa los controladores y NVIDIA Container Toolkit/Docker Desktop.'
        } else {
            Write-Host 'GPU NVIDIA no utilizable desde Docker: se iniciara en CPU.'
        }
    } elseif ($Mode -eq 'auto') {
        Write-Host 'No se detecta GPU NVIDIA en el anfitrion: se iniciara en CPU.'
    }

    if ($useGpu) {
        Write-Host 'Iniciando servicios con GPU NVIDIA...'
        Invoke-Docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
        & docker compose -f compose.yaml -f compose.gpu.yaml exec -T backend python -c 'import torch; assert torch.cuda.is_available(), "CUDA no disponible en PyTorch"; print("CUDA disponible:", torch.cuda.get_device_name(0))'
        if ($LASTEXITCODE -ne 0) {
            if ($Mode -eq 'gpu') {
                throw 'PyTorch no detecta CUDA. Revisa la compatibilidad del controlador con CUDA 13.'
            }
            Write-Host 'PyTorch no detecta CUDA; recreando el backend sin reserva de GPU.'
            Invoke-Docker compose -f compose.yaml up -d --force-recreate backend frontend
            Write-Host 'Modo CPU seleccionado.'
        } else {
            Write-Host 'Modo GPU confirmado dentro del backend.'
        }
    } else {
        Write-Host 'Iniciando servicios en modo CPU...'
        Invoke-Docker compose -f compose.yaml up -d --build
    }
    Write-Host 'Aplicacion: http://localhost:5173'
    Write-Host 'API: http://localhost:8000  |  Estado: docker compose ps'
} catch {
    [Console]::Error.WriteLine("Error: $($_.Exception.Message)")
    exit 1
} finally {
    Pop-Location
}
