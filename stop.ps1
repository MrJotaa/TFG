# PowerShell 5.1+ / 7+. Usage: .\stop.ps1 [stop|uninstall]
# Stops services or removes Compose resources without touching local files.
param(
    [ValidateSet('stop', 'uninstall')]
    [string]$Action = 'stop'
)

$ErrorActionPreference = 'Stop'

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) {
        throw "Docker ha devuelto el codigo ${LASTEXITCODE}: docker $($args -join ' ')"
    }
}

Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath 'compose.yaml')) {
        throw 'No se encuentra compose.yaml junto a stop.ps1.'
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker no esta instalado o no esta en PATH.'
    }
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Se necesita Docker Compose v2.' }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker no responde. Inicia Docker Desktop/Engine.' }

    if ($Action -eq 'stop') {
        Write-Host 'Deteniendo los servicios de la plataforma (se conservan contenedores y datos)...'
        Invoke-Docker compose -f compose.yaml stop
        Write-Host 'Servicios detenidos. Se conservan el volumen de MongoDB y los archivos locales.'
        return
    }

    Write-Warning 'Se eliminaran los contenedores, redes, imagenes locales de Compose y el volumen persistente de MongoDB.'
    Write-Warning 'Se perderan las cuentas, el historial y los metadatos guardados en MongoDB.'
    Write-Host 'No se borraran backend/uploads/, example_datasets/, backend/.env ni el codigo fuente.'
    $confirmation = Read-Host 'Escribe DESINSTALAR para confirmar'
    if ($confirmation -cne 'DESINSTALAR') {
        Write-Host 'Desinstalacion cancelada. No se ha eliminado nada.'
        return
    }

    Invoke-Docker compose -f compose.yaml down --volumes --rmi local --remove-orphans
    Write-Host 'Recursos de Docker retirados. Los archivos locales se han conservado.'
} catch {
    [Console]::Error.WriteLine("Error: $($_.Exception.Message)")
    exit 1
} finally {
    Pop-Location
}
