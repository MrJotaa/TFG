export function resolveExperimentTaskType({
    activeModel = null,
    recoveredEvaluation = null,
} = {}) {
    return (
        recoveredEvaluation?.taskType ||
        activeModel?.task_type ||
        'classification'
    )
}
