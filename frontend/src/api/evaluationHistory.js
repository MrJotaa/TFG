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
    epsilon,
    alpha,
    steps,
    seed = 0,
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
            parameters: {
                epsilon: Number(epsilon),
                alpha: Number(alpha),
                steps: Number(steps),
                seed: Number(seed),
            },
            supervised_configuration: supervisedConfiguration,
        }),
    })
}
