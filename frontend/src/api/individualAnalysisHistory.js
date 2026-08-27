import { apiRequest } from './client.js'

export function saveIndividualAnalysis({ taskType, resultId }) {
    return apiRequest('/individual-analyses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            task_type: taskType,
            result_id: resultId,
        }),
    })
}

export function findCompatibleIndividualAnalysis({
    taskType,
    datasetId,
    imageId,
    attackType,
    parameters,
}) {
    return apiRequest('/individual-analyses/compatibility', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            task_type: taskType,
            dataset_id: datasetId,
            image_id: imageId,
            attack_type: attackType,
            parameters,
        }),
    })
}

export function getIndividualAnalysisHistory({ limit = 10, offset = 0, taskType = '' } = {}) {
    const search = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
    })
    if (taskType) search.set('task_type', taskType)
    return apiRequest(`/individual-analyses?${search.toString()}`)
}

export function getStoredIndividualAnalysis(analysisId) {
    return apiRequest(`/individual-analyses/${encodeURIComponent(analysisId)}`)
}

export function deleteIndividualAnalysis(analysisId) {
    return apiRequest(`/individual-analyses/${encodeURIComponent(analysisId)}`, {
        method: 'DELETE',
    })
}
