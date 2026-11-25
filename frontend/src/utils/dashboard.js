export function getExperimentReadiness({
    selectedDataset,
    selectedImage,
    activeModel,
    attackSupported = true,
    attackParameterError = '',
}) {
    const canPredict = Boolean(activeModel && selectedDataset && selectedImage)
    const canEvaluate = Boolean(activeModel && selectedDataset)
    const attackReady = attackSupported && !attackParameterError
    const actions = {
        canPredict,
        canAnalyze: Boolean(canPredict && attackReady),
        canEvaluate: Boolean(canEvaluate && attackReady),
    }

    if (!activeModel) {
        return { ...actions, tone: 'idle', message: 'Activa un modelo para comenzar.' }
    }
    if (!selectedDataset) {
        return { ...actions, tone: 'idle', message: 'Selecciona un conjunto de datos.' }
    }
    if (!attackReady) {
        const reason = !attackSupported
            ? 'Selecciona un ataque compatible con el modelo activo.'
            : attackParameterError
        return {
            ...actions,
            tone: canPredict || attackParameterError ? 'partial' : 'idle',
            message: canPredict ? `Puedes predecir la imagen. Para aplicar un ataque o evaluar el dataset: ${reason}` : reason,
        }
    }
    if (!selectedImage) {
        return {
            ...actions,
            tone: 'partial',
            message: 'Dataset listo para evaluar. Selecciona una imagen para predecirla o atacarla.',
        }
    }
    return { ...actions, tone: 'ready', message: 'Configuración lista para ejecutar.' }
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
