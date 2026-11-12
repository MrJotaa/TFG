import { stableComparisonKey } from './evaluationComparison.js'
import {
    detectionSuccessLabel,
    getDetectionSuccessPresentation,
} from './detectionAttackSuccess.js'

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
    const success = item?.summary?.attack_success
    if (item?.task_type === 'detection' && typeof success === 'boolean') {
        return detectionSuccessLabel(item.summary, success)
    }
    if (success === true) return 'Ataque exitoso'
    if (success === false) return 'Predicción preservada'
    return 'Resultado guardado'
}

function incompatibleComparison(code) {
    return { compatible: false, code }
}

function comparisonContextMismatch(left, right) {
    if (!left || !right) return 'missing_result'
    if (left.analysis_id === right.analysis_id) return 'same_result'
    if (left.task_type !== right.task_type) return 'different_task'
    if (left.task_type === 'detection'
        && getDetectionSuccessPresentation(left.summary).criterion
            !== getDetectionSuccessPresentation(right.summary).criterion) {
        return 'different_metrics_schema_version'
    }
    if (left.dataset_id !== right.dataset_id) return 'different_dataset'
    if (left.dataset_revision !== right.dataset_revision) return 'different_dataset_revision'
    if (left.image_id !== right.image_id) return 'different_image'
    if (left.pipeline_version && right.pipeline_version && left.pipeline_version !== right.pipeline_version) {
        return 'different_pipeline_version'
    }
    return null
}

function implementationRevisionMismatch(left, right, sameModel, sameAttack) {
    if (sameModel && left.model_revision && right.model_revision && left.model_revision !== right.model_revision) {
        return 'different_model_revision'
    }
    if (sameAttack && left.attack_revision && right.attack_revision && left.attack_revision !== right.attack_revision) {
        return 'different_attack_revision'
    }
    return null
}

function controlledIndividualComparison(sameModel, sameAttack, sameParameters) {
    if (!sameModel && sameAttack && sameParameters) {
        return { compatible: true, kind: 'model', label: 'Modelos' }
    }
    if (sameModel && !sameAttack) {
        return { compatible: true, kind: 'attack', label: 'Ataques' }
    }
    if (sameModel && sameAttack && !sameParameters) {
        return { compatible: true, kind: 'attack_configuration', label: 'Configuración' }
    }
    if (sameModel && sameAttack && sameParameters) return incompatibleComparison('identical_configuration')
    return incompatibleComparison('multiple_variables')
}

export function classifyIndividualAnalysisComparison(left, right) {
    const contextMismatch = comparisonContextMismatch(left, right)
    if (contextMismatch) return incompatibleComparison(contextMismatch)

    const sameModel = left.model_id === right.model_id
    const sameAttack = left.attack_type === right.attack_type
    const sameParameters = stableComparisonKey(left.parameters) === stableComparisonKey(right.parameters)
    const revisionMismatch = implementationRevisionMismatch(left, right, sameModel, sameAttack)
    if (revisionMismatch) return incompatibleComparison(revisionMismatch)

    return controlledIndividualComparison(sameModel, sameAttack, sameParameters)
}

export function getCompatibleIndividualAnalysisCandidates(base, items = []) {
    if (base?.artifact_available === false) return []
    return items.filter((item) => item.artifact_available !== false).flatMap((item) => {
        const comparison = classifyIndividualAnalysisComparison(base, item)
        return comparison.compatible ? [{ item, comparison }] : []
    })
}
