import { apiRequest } from './client'

export function detectImage({ datasetId, imageId }) {
    return apiRequest('/detect', {
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

export function attackDetectionImage({
    datasetId,
    imageId,
    attackType,
    parameters,
}) {
    return apiRequest('/detect/attack', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            image_id: imageId,
            attack_type: attackType,
            parameters,
        }),
    })
}
