# PowerShell 5.1+ / PowerShell 7+.
# Prepara e importa datasets supervisados sin instalar Python en Windows.
param(
    [int]$Count = 100,
    [ValidateSet('both', 'classification', 'detection')]
    [string]$Task = 'both',
    [switch]$PrepareOnly,
    [string]$Email,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

function Show-Usage {
    Write-Host @'
Uso: powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1 [opciones]

  -Count N          10, 20, ..., 100 imagenes por dataset (100 por defecto)
  -Task T           both (por defecto), classification o detection
  -PrepareOnly      Descargar y preparar archivos sin importar en la app
  -Email CORREO     Cuenta existente de la plataforma (pide la contrasena)
  -Help             Mostrar esta ayuda

Por defecto descarga e importa Imagenette y COCO con etiquetas y anotaciones.
Antes de importar, crea una cuenta en http://localhost:5173.
Usa el CORREO y la CONTRASENA de esa cuenta, NO tus datos de GitHub.
La contrasena se solicita sin mostrarla y no se guarda en los datasets.

Ejemplos:
  powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1 -Count 20
  powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1 -Task detection -Count 30
  powershell -ExecutionPolicy Bypass -File .\scripts\datasets\prepare_examples.ps1 -PrepareOnly
'@
}

if ($Help) {
    Show-Usage
    return
}
if ($Count -lt 10 -or $Count -gt 100 -or ($Count % 10) -ne 0) {
    [Console]::Error.WriteLine('Error: -Count debe ser multiplo de 10 entre 10 y 100.')
    exit 2
}

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$ComposeFile = Join-Path $Root 'compose.yaml'
$DataDir = Join-Path $Root 'example_datasets'
$ScriptsDir = Join-Path $Root 'scripts\datasets'
$StartScript = Join-Path $Root 'start.ps1'

try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker no esta instalado o no figura en PATH.'
    }
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Se necesita Docker Compose v2.' }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker no responde. Inicia Docker Desktop.' }
    if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
        throw 'No se encontro compose.yaml en la raiz del repositorio.'
    }
    if (-not (Test-Path -LiteralPath $ScriptsDir -PathType Container)) {
        throw 'No se encontro el directorio de scripts de datasets.'
    }
    [void](New-Item -ItemType Directory -Path $DataDir -Force)

    if (-not $PrepareOnly) {
        # Puede fallar si aun no existe backend/.env; en ese caso, start.ps1
        # configura el entorno y levanta los servicios de manera habitual.
        $running = @()
        try {
            $running = @(& docker compose -f $ComposeFile ps --status running --services 2>$null)
            if ($LASTEXITCODE -ne 0) { $running = @() }
        } catch {
            $running = @()
        }
        if ($running -notcontains 'backend') {
            Write-Host 'Iniciando la plataforma antes de importar los datasets...'
            if (-not (Test-Path -LiteralPath $StartScript -PathType Leaf)) {
                throw 'No existe start.ps1 en la raiz del repositorio.'
            }
            & $StartScript
        }
        Write-Host ''
        Write-Host '=== Importacion automatica en la plataforma ==='
        Write-Host 'Se usara una cuenta creada en http://localhost:5173 (NO la de GitHub).'
        if ($Email) {
            Write-Host 'Se ha indicado el correo; solo se solicitara la contrasena.'
        } else {
            Write-Host 'Se solicitaran el correo y la contrasena de esa cuenta.'
        }
        Write-Host 'La contrasena no se mostrara al escribirla.'
        Write-Host 'Las etiquetas de clasificacion y anotaciones COCO se importan automaticamente.'
        Write-Host ''
    } else {
        Write-Host 'Solo preparacion: no se solicitan credenciales ni se importa en la app.'
    }

    # Se ejecuta el MISMO Python que usa el lanzador .sh, dentro del backend.
    # Cada volumen se pasa como argumento unico; funciona con rutas C:\... y espacios.
    Write-Host 'Preparando la imagen Docker del backend...'
    & docker compose -f $ComposeFile build backend
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo construir la imagen del backend.' }

    $scriptsMount = '{0}:/opt/tfg-datasets:ro' -f $ScriptsDir
    $dataMount = '{0}:/datasets' -f $DataDir
    $runArgs = @('compose', '-f', $ComposeFile, 'run', '--rm', '--no-deps')
    if ($PrepareOnly) { $runArgs += '-T' }
    $dockerArgs = $runArgs + @(
        '-v', $scriptsMount, '-v', $dataMount,
        'backend', 'python', '/opt/tfg-datasets/prepare_examples.py',
        '--count', [string]$Count, '--task', $Task,
        '--data-dir', '/datasets', '--api-base', 'http://backend:8000'
    )
    if ($PrepareOnly) { $dockerArgs += '--prepare-only' }
    if ($Email) { $dockerArgs += @('--email', $Email) }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'La preparacion o importacion no se completo. Repite el mismo comando para continuar.'
    }
} catch {
    [Console]::Error.WriteLine("Error: $($_.Exception.Message)")
    exit 1
}
