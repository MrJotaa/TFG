export function getExperimentReadiness({
    selectedDataset,
    selectedImage,
    activeModel,
    attackSupported = true,
    attackParameterError = '',
}) {
    if (!activeModel) {
        return {
            canAnalyze: false,
            canEvaluate: false,
            tone: 'idle',
            message: 'Activa un modelo para comenzar.',
        }
    }

    if (!attackSupported) {
        return {
            canAnalyze: false,
            canEvaluate: false,
            tone: 'idle',
            message: 'Selecciona un ataque compatible con el modelo activo.',
        }
    }

    if (attackParameterError) {
        return {
            canAnalyze: false,
            canEvaluate: false,
            tone: 'partial',
            message: attackParameterError,
        }
    }

    const canEvaluate = Boolean(selectedDataset)
    const canAnalyze = Boolean(canEvaluate && selectedImage)

    if (!selectedDataset) {
        return {
            canAnalyze,
            canEvaluate,
            tone: 'idle',
            message: 'Selecciona un conjunto de datos.',
        }
    }

    if (!selectedImage) {
        return {
            canAnalyze,
            canEvaluate,
            tone: 'partial',
            message: 'Dataset listo. Selecciona una imagen para el análisis individual.',
        }
    }

    return {
        canAnalyze,
        canEvaluate,
        tone: 'ready',
        message: 'Configuración lista para ejecutar.',
    }
}


export function resolveScrollOverflow({
    scrollTop,
    scrollHeight,
    clientHeight,
    tolerance = 1,
}) {
    const top = Math.max(0, Number(scrollTop) || 0)
    const height = Math.max(0, Number(scrollHeight) || 0)
    const viewport = Math.max(0, Number(clientHeight) || 0)
    const maxScrollTop = Math.max(0, height - viewport)

    return {
        canScrollUp: top > tolerance,
        canScrollDown: maxScrollTop - top > tolerance,
    }
}

export function clampDashboardSidebarWidth(width) {
    return Math.min(Math.max(width, 275), 430)
}

export function resolveDashboardSidebarWidthKey(key, currentWidth) {
    if (key === 'Home') return 275
    if (key === 'End') return 430
    if (key === 'ArrowLeft') return clampDashboardSidebarWidth(currentWidth - 10)
    if (key === 'ArrowRight') return clampDashboardSidebarWidth(currentWidth + 10)
    return null
}
