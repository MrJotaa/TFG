import { apiRequest } from './client'

export function predictImage({ datasetId, imageId, signal }) {
    return apiRequest('/predict', {
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
