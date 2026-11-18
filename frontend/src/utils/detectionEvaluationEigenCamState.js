export const IDLE_DETECTION_EIGENCAM_STATE = Object.freeze({
    status: 'idle',
    data: null,
    error: '',
})

export function getDetectionEigenCamKey(evaluationId, imageId) {
    return JSON.stringify([evaluationId, imageId])
}

export function getDetectionEigenCamEntry(state, key) {
    return state[key] ?? IDLE_DETECTION_EIGENCAM_STATE
}

export function detectionEigenCamReducer(state, action) {
    if (action.type === 'reset') {
        return {}
    }

    if (action.type === 'loading') {
        return {
            ...state,
            [action.key]: { status: 'loading', data: null, error: '' },
        }
    }

    if (action.type === 'success') {
        return {
            ...state,
            [action.key]: { status: 'success', data: action.data, error: '' },
        }
    }

    if (action.type === 'error') {
        return {
            ...state,
            [action.key]: { status: 'error', data: null, error: action.error },
        }
    }

    return state
}

export function getDetectionEigenCamButtonLabel(entry, isVisible = true) {
    if (entry?.status === 'loading') return 'Generando Eigen-CAM…'
    if (entry?.status === 'success') return isVisible ? 'Ocultar Eigen-CAM' : 'Mostrar Eigen-CAM'
    if (entry?.status === 'error') return 'Reintentar Eigen-CAM'
    return 'Generar Eigen-CAM'
}

export function canRequestDetectionEigenCam(entry, isInFlight = false) {
    return (
        !isInFlight
        && entry.status !== 'loading'
        && entry.status !== 'success'
    )
}

export function isDetectionEigenCamResponse(response, evaluationId, imageId) {
    return Boolean(
        response
        && response.evaluation_id === evaluationId
        && response.image_id === imageId
        && response.explanation_type === 'eigencam'
        && response.original_eigencam_image_url
        && response.adversarial_eigencam_image_url,
    )
}

export function getDetectionEigenCamErrorMessage(error) {
    const messages = {
        400: 'No se puede generar Eigen-CAM: los datos no son válidos o el modelo activo no coincide con la evaluación.',
        401: 'La sesión ha caducado. Inicia sesión de nuevo para generar Eigen-CAM.',
        403: 'No tienes permiso para acceder a esta evaluación.',
        404: 'La evaluación o la imagen solicitada ya no existe.',
        410: 'La imagen temporal tras el ataque ya no está disponible. Repite la evaluación para generar Eigen-CAM.',
        500: 'No se pudo generar o guardar Eigen-CAM. Puedes intentarlo de nuevo.',
    }

    return (
        messages[error?.status]
        ?? 'No se pudo conectar con el servidor para generar Eigen-CAM. Comprueba la conexión e inténtalo de nuevo.'
    )
}
