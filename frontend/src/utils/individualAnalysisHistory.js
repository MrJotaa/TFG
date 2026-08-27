import { stableComparisonKey } from './evaluationComparison.js'

export function individualAnalysisSummary(item) {
    const summary = item?.summary ?? {}

    if (item?.task_type === 'detection') {
        const original = summary.original_detection_count ?? 0
        const adversarial = summary.adversarial_detection_count ?? 0
        return `${original} → ${adversarial} detecciones`
    }

    const original = summary.original_label || '—'
    const adversarial = summary.adversarial_label || '—'
    return `${original} → ${adversarial}`
}

export function individualAnalysisOutcome(item) {
    if (item?.summary?.attack_success === true) return 'Ataque exitoso'
    if (item?.summary?.attack_success === false) {
        return item?.task_type === 'detection'
            ? 'Detecciones preservadas'
            : 'Predicción preservada'
    }
    return 'Resultado guardado'
}

export function classifyIndividualAnalysisComparison(left, right) {
    if (!left || !right) return { compatible: false, code: 'missing_result' }
    if (left.analysis_id === right.analysis_id) {
        return { compatible: false, code: 'same_result' }
    }
    if (left.task_type !== right.task_type) {
        return { compatible: false, code: 'different_task' }
    }
    if (left.dataset_id !== right.dataset_id) {
        return { compatible: false, code: 'different_dataset' }
    }
    if (left.dataset_revision !== right.dataset_revision) {
        return { compatible: false, code: 'different_dataset_revision' }
    }
    if (left.image_id !== right.image_id) {
        return { compatible: false, code: 'different_image' }
    }

    if (
        left.pipeline_version
        && right.pipeline_version
        && left.pipeline_version !== right.pipeline_version
    ) {
        return { compatible: false, code: 'different_pipeline_version' }
    }

    const sameModel = left.model_id === right.model_id
    const sameAttack = left.attack_type === right.attack_type
    const sameParameters = stableComparisonKey(left.parameters)
        === stableComparisonKey(right.parameters)

    if (
        sameModel
        && left.model_revision
        && right.model_revision
        && left.model_revision !== right.model_revision
    ) {
        return { compatible: false, code: 'different_model_revision' }
    }
    if (
        sameAttack
        && left.attack_revision
        && right.attack_revision
        && left.attack_revision !== right.attack_revision
    ) {
        return { compatible: false, code: 'different_attack_revision' }
    }

    if (!sameModel && sameAttack && sameParameters) {
        return {
            compatible: true,
            kind: 'model',
            label: 'Modelos',
        }
    }
    if (sameModel && !sameAttack) {
        return {
            compatible: true,
            kind: 'attack',
            label: 'Ataques',
        }
    }
    if (sameModel && sameAttack && !sameParameters) {
        return {
            compatible: true,
            kind: 'attack_configuration',
            label: 'Configuración',
        }
    }
    if (sameModel && sameAttack && sameParameters) {
        return { compatible: false, code: 'identical_configuration' }
    }
    return { compatible: false, code: 'multiple_variables' }
}

export function getCompatibleIndividualAnalysisCandidates(base, items = []) {
    return items.flatMap((item) => {
        const comparison = classifyIndividualAnalysisComparison(base, item)
        return comparison.compatible ? [{ item, comparison }] : []
    })
}
