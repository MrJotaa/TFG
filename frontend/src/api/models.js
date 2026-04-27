import { apiRequest } from './client'

export function getModels() {
    return apiRequest('/models')
}

export function getActiveModel() {
    return apiRequest('/models/active')
}

export function selectModel(modelId) {
    return apiRequest(`/models/${modelId}/select`, {
        method: 'POST',
    })
}