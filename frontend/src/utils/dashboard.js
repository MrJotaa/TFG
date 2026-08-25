export function getExperimentReadiness({
    selectedDataset,
    selectedImage,
    activeModel,
    attackSupported = true,
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
