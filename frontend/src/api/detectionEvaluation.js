import { apiRequest } from './client.js'

export function evaluateDetectionDataset({
    datasetId,
    attackType,
    epsilon,
    alpha,
    steps,
    supervisedConfiguration,
    seed = 0,
    forceRecompute = false,
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
                seed,
            },
            supervised_configuration: supervisedConfiguration,
            force_recompute: forceRecompute,
        }),
    })
}

export function generateDetectionEvaluationEigenCam(evaluationId, imageId) {
    const encodedEvaluationId = encodeURIComponent(evaluationId)
    const encodedImageId = encodeURIComponent(imageId)

    return apiRequest(
        `/detect/evaluations/${encodedEvaluationId}/images/${encodedImageId}/eigencam`,
        {
            method: 'POST',
        },
    )
}
