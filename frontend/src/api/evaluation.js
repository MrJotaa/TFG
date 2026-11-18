import { apiRequest } from './client.js'
import { streamEvaluationRequest } from './evaluationStream.js'

export function evaluateDataset({
    datasetId,
    attackType,
    parameters,
    forceRecompute = false,
    onProgress,
    signal,
}) {
    return streamEvaluationRequest(
        '/evaluate/stream',
        {
            dataset_id: datasetId,
            attack_type: attackType,
            parameters,
            force_recompute: forceRecompute,
        },
        { onProgress, signal },
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
