import { apiRequest } from './client.js'
import { streamEvaluationRequest } from './evaluationStream.js'

export function evaluateDataset({
    datasetId,
    attackType,
    epsilon,
    alpha,
    steps,
    seed = 0,
    forceRecompute = false,
    onProgress,
}) {
    return streamEvaluationRequest(
        '/evaluate/stream',
        {
            dataset_id: datasetId,
            attack_type: attackType,
            parameters: {
                epsilon: Number(epsilon),
                alpha: Number(alpha),
                steps: Number(steps),
                seed: Number(seed),
            },
            force_recompute: forceRecompute,
        },
        { onProgress },
    )
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
