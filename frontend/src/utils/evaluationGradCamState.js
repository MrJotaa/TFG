export const IDLE_EVALUATION_GRADCAM_STATE = Object.freeze({
    status: 'idle',
    data: null,
    error: '',
})

export function getEvaluationGradCamKey(evaluationId, imageId) {
    return JSON.stringify([evaluationId, imageId])
}

export function getEvaluationGradCamEntry(state, key) {
    return state[key] ?? IDLE_EVALUATION_GRADCAM_STATE
}

export function evaluationGradCamReducer(state, action) {
    if (action.type === 'reset') {
        return {}
    }

    if (action.type === 'loading') {
        return {
            ...state,
            [action.key]: {
                status: 'loading',
                data: null,
                error: '',
            },
        }
    }

    if (action.type === 'success') {
        return {
            ...state,
            [action.key]: {
                status: 'success',
                data: action.data,
                error: '',
            },
        }
    }

    if (action.type === 'error') {
        return {
            ...state,
            [action.key]: {
                status: 'error',
                data: null,
                error: action.error,
            },
        }
    }

    return state
}

export function getEvaluationGradCamButtonLabel(entry, isVisible = true) {
    if (entry?.status === 'loading') return 'Generando Grad-CAM…'
    if (entry?.status === 'success') return isVisible ? 'Ocultar Grad-CAM' : 'Mostrar Grad-CAM'
    if (entry?.status === 'error') return 'Reintentar Grad-CAM'
    return 'Generar Grad-CAM'
}

export function canRequestEvaluationGradCam(entry, isInFlight = false) {
    return (
        !isInFlight
        && entry.status !== 'loading'
        && entry.status !== 'success'
    )
}

export function isEvaluationGradCamResponse(response, evaluationId, imageId) {
    return Boolean(
        response
        && response.evaluation_id === evaluationId
        && response.image_id === imageId
        && response.original_gradcam_image_url
        && response.adversarial_gradcam_image_url,
    )
}

export function getEvaluationGradCamErrorMessage(error) {
    const messages = {
        400: 'No se puede generar Grad-CAM: la evaluación está incompleta, sus datos no son válidos o el modelo activo no coincide.',
        401: 'La sesión ha caducado. Inicia sesión de nuevo para generar Grad-CAM.',
        403: 'No tienes permiso para acceder a esta evaluación.',
        404: 'La evaluación o la imagen solicitada ya no existe.',
        410: 'Los artefactos temporales necesarios ya no están disponibles. Repite la evaluación para generar Grad-CAM.',
        500: 'No se pudo generar o guardar Grad-CAM. Puedes intentarlo de nuevo.',
    }

    return (
        messages[error?.status]
        ?? 'No se pudo conectar con el servidor para generar Grad-CAM. Comprueba la conexión e inténtalo de nuevo.'
    )
}
