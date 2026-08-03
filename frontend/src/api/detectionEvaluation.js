import { apiRequest } from './client'

export function evaluateDetectionDataset({
    datasetId,
    attackType,
    epsilon,
    alpha,
    steps,
    supervisedConfiguration,
}) {
    return apiRequest('/detect/evaluate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: datasetId,
            attack_type: attackType,
            parameters: {
                epsilon,
                alpha,
                steps,
            },
            supervised_configuration: supervisedConfiguration,
        }),
    })
}
