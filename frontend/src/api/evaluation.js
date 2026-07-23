import { apiRequest } from './client.js'

export function evaluateDataset({ datasetId, attackType, epsilon, alpha, steps }) {
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
                alpha: Number(alpha),
                steps: Number(steps),
            },
        }),
    })
}

export function generateEvaluationGradCam(evaluationId, imageId) {
    const encodedEvaluationId = encodeURIComponent(evaluationId)
    const encodedImageId = encodeURIComponent(imageId)

    return apiRequest(
        `/evaluations/${encodedEvaluationId}/images/${encodedImageId}/gradcam`,
        {
            method: 'POST',
        },
    )
}
