import { apiRequest } from './client.js'

export function getEvaluationHistory({ limit = 10, offset = 0, taskType = '' } = {}) {
    const search = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
    })
    if (taskType) search.set('task_type', taskType)
    return apiRequest(`/evaluations?${search.toString()}`)
}

export async function getAllEvaluationHistory({ taskType = '' } = {}) {
    const limit = 50
    let offset = 0
    let total
    const items = []

    do {
        const page = await getEvaluationHistory({ limit, offset, taskType })
        items.push(...(page.items ?? []))
        total = page.total ?? items.length
        offset += limit
    } while (offset < total)

    return items
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

export function setEvaluationPreserved(evaluationId, preserved) {
    return apiRequest(`/evaluations/${encodeURIComponent(evaluationId)}/preservation`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preserved: Boolean(preserved) }),
    })
}

export function deleteEvaluation(evaluationId) {
    return apiRequest(`/evaluations/${encodeURIComponent(evaluationId)}`, {
        method: 'DELETE',
    })
}
