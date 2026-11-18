import { apiRequest } from './client.js'
import { streamEvaluationRequest } from './evaluationStream.js'

export function evaluateDetectionDataset({
    datasetId,
    attackType,
    parameters,
    supervisedConfiguration,
    forceRecompute = false,
    onProgress,
    signal,
}) {
    return streamEvaluationRequest(
        '/detect/evaluate/stream',
        {
            dataset_id: datasetId,
            attack_type: attackType,
            parameters,
            supervised_configuration: supervisedConfiguration,
            force_recompute: forceRecompute,
        },
        { onProgress, signal },
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
