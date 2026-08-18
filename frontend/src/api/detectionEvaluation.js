import { apiRequest } from './client.js'
import { streamEvaluationRequest } from './evaluationStream.js'

export function evaluateDetectionDataset({
    datasetId,
    attackType,
    epsilon,
    alpha,
    steps,
    supervisedConfiguration,
    seed = 0,
    forceRecompute = false,
    onProgress,
}) {
    return streamEvaluationRequest(
        '/detect/evaluate/stream',
        {
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
        },
        { onProgress },
    )
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
