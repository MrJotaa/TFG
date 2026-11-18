import { apiRequest } from './client'

export function generateGradCam({ datasetId, imageId, signal }) {
    return apiRequest('/explain/gradcam', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            image_id: imageId,
        }),
        signal,
    })
}

export function generateDetectionExplainability({ datasetId, imageId, signal }) {
    return apiRequest('/explain/detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            image_id: imageId,
        }),
        signal,
    })
}
