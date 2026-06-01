import { apiRequest } from './client'

export function evaluateDataset({ datasetId, attackType, epsilon }) {
    return apiRequest('/evaluate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            attack_type: attackType,
            parameters: {
                epsilon: Number(epsilon),
            },
        }),
    })
}