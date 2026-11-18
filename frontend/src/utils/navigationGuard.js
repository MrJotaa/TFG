export function shouldProtectNavigation({
    hasTemporaryWork = false,
    busy = false,
} = {}) {
    return Boolean(hasTemporaryWork || busy)
}

export function getNavigationGuardDialogCopy({
    busy = false,
    intent = 'navigate',
} = {}) {
    if (intent === 'logout') {
        return {
            title: 'Cerrar sesión',
            message: busy
                ? 'Hay un análisis en curso. Si cierras sesión, se cancelará y los resultados temporales no guardados se perderán.'
                : 'Hay resultados temporales de análisis que todavía no se han guardado. Si cierras sesión, se perderán.',
            confirmLabel: 'Cerrar sesión',
            cancelLabel: 'Seguir analizando',
            tone: 'danger',
        }
    }

    if (busy) {
        return {
            title: 'Análisis en curso',
            message: 'Hay una operación en curso. Si abandonas esta página, se cancelará y los resultados temporales no guardados se perderán.',
            confirmLabel: 'Salir y cancelar análisis',
            cancelLabel: 'Seguir analizando',
            tone: 'warning',
        }
    }

    return {
        title: 'Resultados temporales sin guardar',
        message: 'Si abandonas esta página, los resultados del análisis individual se perderán. Puedes seguir trabajando o descartarlos y salir.',
        confirmLabel: 'Salir y descartar',
        cancelLabel: 'Seguir analizando',
        tone: 'warning',
    }
}

export function protectBeforeUnload(event) {
    if (!event) return false

    event.preventDefault?.()
    event.returnValue = ''
    return true
}
