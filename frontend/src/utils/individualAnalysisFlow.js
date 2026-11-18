import { stableComparisonKey } from './evaluationComparison.js'
import { classifyIndividualAnalysisComparison } from './individualAnalysisHistory.js'

const KIND_COPY = {
    model: {
        label: 'Otro modelo',
        instruction: 'Mantén la misma imagen, ataque y parámetros; activa un modelo diferente.',
    },
    attack: {
        label: 'Otro ataque',
        instruction: 'Mantén la misma imagen y modelo; selecciona un ataque diferente.',
    },
    attack_configuration: {
        label: 'Otra configuración',
        instruction: 'Mantén el mismo modelo y ataque; cambia al menos un parámetro.',
    },
}

const INVALID_COMPARISON_MESSAGES = {
    different_task: 'El modelo activo pertenece a otra tarea.',
    different_dataset: 'La comparación debe usar el mismo dataset.',
    different_dataset_revision: 'El dataset ha cambiado desde el análisis A. Repite A sobre la revisión actual antes de comparar.',
    different_image: 'La comparación debe usar exactamente la misma imagen.',
    different_pipeline_version: 'Los análisis usan versiones diferentes del pipeline.',
    different_model_revision: 'La revisión interna del modelo ya no coincide con la del análisis A.',
    different_attack_revision: 'La revisión interna del ataque ya no coincide con la del análisis A.',
    identical_configuration: 'La configuración B sigue siendo idéntica a A.',
    multiple_variables: 'Has cambiado más de una variable experimental. Mantén la comparación controlada.',
}

export function individualComparisonKindCopy(kind) {
    return KIND_COPY[kind] ?? KIND_COPY.attack
}

export function makeTemporaryAnalysisItem({ identity, analysis }) {
    const resultId = analysis?.result_id
    if (!identity || !resultId) return null

    return {
        analysis_id: `temporary:${resultId}`,
        source_result_id: resultId,
        task_type: identity.task_type,
        dataset_id: identity.dataset_id,
        dataset_name: identity.dataset_name,
        dataset_revision: identity.dataset_revision,
        image_id: identity.image_id,
        image_filename: identity.image_filename,
        model_id: identity.model_id,
        model_name: identity.model_name,
        model_revision: identity.model_revision ?? null,
        attack_type: identity.attack_type,
        attack_revision: identity.attack_revision ?? null,
        parameters: identity.parameters ?? {},
        pipeline_version: identity.pipeline_version ?? null,
        explanation_available: false,
        created_at: null,
        saved_at: null,
    }
}

export function makeAnalysisReference({
    item,
    analysis,
    dataset,
    image,
    temporary = false,
}) {
    return {
        item,
        analysis,
        dataset,
        image,
        temporary: Boolean(temporary),
        resultId: analysis?.result_id ?? item?.source_result_id ?? null,
    }
}

export function resolveSaveState(currentResultId, savedItem) {
    if (!currentResultId || !savedItem) return 'temporary'
    return savedItem.source_result_id === currentResultId
        ? 'saved'
        : 'equivalent'
}

export function validatePreparedComparison({ baseItem, expectedKind, identity }) {
    const candidate = {
        ...identity,
        analysis_id: `candidate:${identity?.model_id ?? 'model'}:${identity?.attack_type ?? 'attack'}`,
    }
    const comparison = classifyIndividualAnalysisComparison(baseItem, candidate)
    if (!comparison.compatible) {
        return {
            valid: false,
            comparison,
            message: INVALID_COMPARISON_MESSAGES[comparison.code]
                ?? 'La configuración B no es comparable con A.',
        }
    }
    if (comparison.kind !== expectedKind) {
        return {
            valid: false,
            comparison,
            message: `La configuración actual corresponde a «${comparison.label}», pero estás preparando «${individualComparisonKindCopy(expectedKind).label}».`,
        }
    }
    return { valid: true, comparison, message: '' }
}

function notReady(message) {
    return { ready: false, message }
}

function comparisonContextReadiness({ baseItem, selectedDataset, selectedImage, activeModel }) {
    if (!baseItem) return notReady('Falta el análisis A.')
    if (selectedDataset?.id !== baseItem.dataset_id) return notReady('Vuelve al dataset usado por A.')
    if (
        selectedDataset?.dataset_revision != null
        && Number(selectedDataset.dataset_revision) !== Number(baseItem.dataset_revision)
    ) {
        return notReady('El dataset ha cambiado desde A. La comparación requiere la misma revisión.')
    }
    if (selectedImage?.id !== baseItem.image_id) return notReady('Selecciona la misma imagen usada por A.')
    if (!activeModel) return notReady('Activa un modelo para preparar B.')
    if (activeModel.task_type !== baseItem.task_type) return notReady('El modelo activo pertenece a otra tarea.')
    return null
}

function modelComparisonReadiness({ sameModel, sameAttack, sameParameters }) {
    if (sameModel) return notReady('Activa un modelo diferente para B.')
    if (!sameAttack || !sameParameters) {
        return notReady('Para comparar modelos debes conservar el ataque y sus parámetros de A.')
    }
    return null
}

function attackComparisonReadiness({ baseItem, sameModel, sameAttack }) {
    if (!sameModel) return notReady(`Activa de nuevo ${baseItem.model_name} para comparar ataques.`)
    if (sameAttack) return notReady('Selecciona un ataque diferente para B.')
    return null
}

function attackConfigurationReadiness({ baseItem, sameModel, sameAttack, sameParameters }) {
    if (!sameModel) return notReady(`Activa de nuevo ${baseItem.model_name} para comparar configuraciones.`)
    if (!sameAttack) return notReady('Mantén el mismo ataque de A para comparar configuraciones.')
    if (sameParameters) return notReady('Modifica al menos un parámetro del ataque para B.')
    return null
}

function comparisonKindReadiness(expectedKind, state) {
    if (expectedKind === 'model') return modelComparisonReadiness(state)
    if (expectedKind === 'attack') return attackComparisonReadiness(state)
    if (expectedKind === 'attack_configuration') return attackConfigurationReadiness(state)
    return null
}

export function pendingComparisonReadiness({
    baseItem,
    expectedKind,
    selectedDataset,
    selectedImage,
    activeModel,
    selectedAttack,
    attackParameters,
}) {
    const contextIssue = comparisonContextReadiness({
        baseItem,
        selectedDataset,
        selectedImage,
        activeModel,
    })
    if (contextIssue) return contextIssue

    const state = {
        baseItem,
        sameModel: activeModel.id === baseItem.model_id,
        sameAttack: selectedAttack === baseItem.attack_type,
        sameParameters: stableComparisonKey(attackParameters) === stableComparisonKey(baseItem.parameters),
    }
    const kindIssue = comparisonKindReadiness(expectedKind, state)
    if (kindIssue) return kindIssue

    return {
        ready: true,
        message: `B está lista. ${individualComparisonKindCopy(expectedKind).instruction}`,
    }
}
