import { apiRequest } from './client.js'

export function getEvaluationHistory({ limit = 10, offset = 0, taskType = '' } = {}) {
    const search = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
    })
    if (taskType) search.set('task_type', taskType)
    return apiRequest(`/evaluations?${search.toString()}`)
}

export function getStoredEvaluation(evaluationId) {
    return apiRequest(`/evaluations/${encodeURIComponent(evaluationId)}`)
}

export function findCompatibleEvaluation({
    taskType,
    datasetId,
    attackType,
    parameters,
    supervisedConfiguration = null,
}) {
    return apiRequest('/evaluations/compatibility', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            task_type: taskType,
            dataset_id: datasetId,
            attack_type: attackType,
            parameters,
            supervised_configuration: supervisedConfiguration,
        }),
    })
}
