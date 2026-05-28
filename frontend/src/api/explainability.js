import { apiRequest } from './client'

export function generateGradCam({ datasetId, imageId }) {
    return apiRequest('/explain/gradcam', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            image_id: imageId,
        }),
    })
}