import { apiRequest } from './client'

export function getDatasets() {
    return apiRequest('/datasets')
}

export function getDatasetById(datasetId) {
    return apiRequest(`/datasets/${datasetId}`)
}

export function getDatasetImages(datasetId) {
    return apiRequest(`/datasets/${datasetId}/images`)
}

export function uploadDataset({ userId, name, files }) {
    const formData = new FormData()

    formData.append('user_id', userId)
    formData.append('name', name)

    Array.from(files).forEach((file) => {
        formData.append('files', file)
    })

    return apiRequest('/datasets', {
        method: 'POST',
        body: formData,
    })
}

export function getDatasetsByUser(userId) {
    return apiRequest(`/users/${userId}/datasets`)
}

export function addImagesToDataset({ datasetId, files }) {
    const formData = new FormData()

    Array.from(files).forEach((file) => {
        formData.append('files', file)
    })

    return apiRequest(`/datasets/${datasetId}/images`, {
        method: 'POST',
        body: formData,
    })
}

export function updateImageLabel({ datasetId, imageId, label }) {
    return apiRequest(`/datasets/${datasetId}/images/${imageId}/label`, {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            label,
        }),
    })
}