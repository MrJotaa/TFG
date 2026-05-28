import { apiRequest } from './client'

export function applyAttack({ datasetId, imageId, attackType, epsilon }) {
    return apiRequest('/attack', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            image_id: imageId,
            attack_type: attackType,
            parameters: {
                epsilon,
            },
        }),
    })
}