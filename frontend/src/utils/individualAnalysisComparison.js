import { classifyIndividualAnalysisComparison } from './individualAnalysisHistory.js'

const COMPARISON_DESCRIPTIONS = {
    model: 'Misma imagen y mismo ataque; cambia únicamente el modelo.',
    attack: 'Misma imagen y mismo modelo; cambia únicamente el ataque.',
    attack_configuration: 'Misma imagen, modelo y ataque; cambian únicamente sus parámetros.',
}

const INCOMPATIBILITY_REASONS = {
    missing_result: 'Falta uno de los análisis seleccionados.',
    same_result: 'Selecciona dos análisis diferentes.',
    different_task: 'Los análisis pertenecen a tareas diferentes.',
    different_dataset: 'Los análisis pertenecen a datasets diferentes.',
    different_dataset_revision: 'Los análisis usan revisiones diferentes del dataset.',
    different_image: 'Los análisis no corresponden a la misma imagen.',
    different_pipeline_version: 'Los análisis usan versiones diferentes del pipeline.',
    different_model_revision: 'La revisión del modelo no coincide.',
    different_attack_revision: 'La revisión del ataque no coincide.',
    identical_configuration: 'Los dos análisis tienen la misma configuración.',
    multiple_variables: 'La pareja cambia más de una variable experimental.',
}

export function buildIndividualAnalysisComparison(leftItem, rightItem) {
    const comparison = classifyIndividualAnalysisComparison(leftItem, rightItem)
    if (!comparison.compatible) {
        return {
            ...comparison,
            reason: INCOMPATIBILITY_REASONS[comparison.code]
                ?? 'Los análisis seleccionados no son comparables.',
        }
    }
    return {
        ...comparison,
        description: COMPARISON_DESCRIPTIONS[comparison.kind],
    }
}


export function comparisonExplanationState(taskType, leftAnalysis, rightAnalysis) {
    const left = taskType === 'detection'
        ? Boolean(
            leftAnalysis?.original_explanation_image_url
            && leftAnalysis?.adversarial_explanation_image_url,
        )
        : Boolean(
            leftAnalysis?.original_gradcam_image_url
            && leftAnalysis?.adversarial_gradcam_image_url,
        )
    const right = taskType === 'detection'
        ? Boolean(
            rightAnalysis?.original_explanation_image_url
            && rightAnalysis?.adversarial_explanation_image_url,
        )
        : Boolean(
            rightAnalysis?.original_gradcam_image_url
            && rightAnalysis?.adversarial_gradcam_image_url,
        )

    return {
        left,
        right,
        comparable: left && right,
    }
}

function resolveComparisonExplanationSource(taskType, analysis, stage) {
    if (taskType === 'detection') {
        return stage === 'original'
            ? analysis.original_explanation_image_url ?? null
            : analysis.adversarial_explanation_image_url ?? null
    }
    return stage === 'original'
        ? analysis.original_gradcam_image_url ?? null
        : analysis.adversarial_gradcam_image_url ?? null
}

function resolveDetectionComparisonImageSource(analysis, image, stage, showAnnotations) {
    if (stage === 'original') {
        return showAnnotations
            ? analysis.original_annotated_image_url ?? image?.url ?? null
            : image?.url ?? null
    }
    return showAnnotations
        ? analysis.adversarial_annotated_image_url ?? analysis.adversarial_image_url ?? null
        : analysis.adversarial_image_url ?? null
}

export function resolveIndividualAnalysisComparisonSource({
    taskType,
    analysis,
    image,
    layer = 'image',
    stage = 'adversarial',
    showAnnotations = true,
}) {
    if (!analysis) return null
    if (layer === 'perturbation') return analysis.perturbation_image_url ?? null
    if (layer === 'explanation') {
        return resolveComparisonExplanationSource(taskType, analysis, stage)
    }
    if (taskType === 'detection') {
        return resolveDetectionComparisonImageSource(analysis, image, stage, showAnnotations)
    }
    return stage === 'original' ? image?.url ?? null : analysis.adversarial_image_url ?? null
}

export function individualAnalysisComparisonSummary(taskType, analysis) {
    if (!analysis) return null

    if (taskType === 'detection') {
        const comparison = analysis.comparison ?? {}
        return {
            primaryLabel: `${analysis.original_detections?.length ?? 0} → ${analysis.adversarial_detections?.length ?? 0}`,
            primaryCaption: 'detecciones',
            originalConfidence: comparison.average_original_confidence,
            adversarialConfidence: comparison.average_adversarial_confidence,
            confidenceDifference: Number.isFinite(Number(comparison.average_confidence_difference))
                ? -Number(comparison.average_confidence_difference)
                : null,
            attackSuccess: comparison.attack_success,
            perturbationLinf: comparison.perturbation_linf,
            perturbationL2: comparison.perturbation_l2,
        }
    }

    const comparison = analysis.comparison ?? {}
    return {
        primaryLabel: `${analysis.original_prediction?.prediction ?? '—'} → ${analysis.adversarial_prediction?.prediction ?? '—'}`,
        primaryCaption: comparison.prediction_changed ? 'predicción modificada' : 'predicción preservada',
        originalConfidence: comparison.original_confidence,
        adversarialConfidence: comparison.adversarial_confidence,
        confidenceDifference: comparison.confidence_difference,
        originalTopK: analysis.original_prediction?.top_k ?? [],
        adversarialTopK: analysis.adversarial_prediction?.top_k ?? [],
        attackSuccess: comparison.attack_success,
        perturbationLinf: comparison.perturbation_linf,
        perturbationL2: comparison.perturbation_l2,
    }
}

export function comparisonLayerOptions(taskType, leftAnalysis, rightAnalysis) {
    const explanation = comparisonExplanationState(taskType, leftAnalysis, rightAnalysis)
    const options = [
        { id: 'image', label: 'Imagen' },
        { id: 'perturbation', label: 'Perturbación' },
    ]
    if (explanation.comparable) {
        options.push({
            id: 'explanation',
            label: taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM',
        })
    }
    return options
}

export function formatComparisonMagnitude(value) {
    const numeric = Number(value)
    if (!Number.isFinite(numeric)) return '—'
    if (numeric === 0) return '0'
    if (Math.abs(numeric) < 0.001) return numeric.toExponential(2)

    const fixed = numeric.toFixed(4)
    let end = fixed.length
    while (end > 0 && fixed[end - 1] === '0') end -= 1
    if (end > 0 && fixed[end - 1] === '.') end -= 1
    return fixed.slice(0, end)
}


function buildTransientComparisonState(transientPair) {
    const left = transientPair?.left
    const right = transientPair?.right
    if (!left?.item || !right?.item || !left?.analysis || !right?.analysis) {
        throw new Error('La comparación temporal ya no contiene los dos resultados necesarios.')
    }

    const comparison = buildIndividualAnalysisComparison(left.item, right.item)
    if (!comparison.compatible) throw new Error(comparison.reason)

    const dataset = transientPair.dataset ?? left.dataset ?? right.dataset
    const image = transientPair.image ?? left.image ?? right.image
    if (!dataset || !image) {
        throw new Error('No se pudo recuperar el contexto visual de la imagen comparada.')
    }

    return { status: 'ready', left, right, comparison, dataset, image }
}

function buildSavedReference(stored, dataset, image) {
    return {
        item: stored.item,
        analysis: stored.analysis,
        dataset,
        image,
        temporary: false,
        resultId: stored.analysis?.result_id ?? stored.item?.source_result_id ?? null,
    }
}

async function buildStoredComparisonState(leftId, rightId, dependencies) {
    if (!leftId || !rightId || leftId === rightId) {
        throw new Error('Selecciona dos análisis individuales diferentes desde el historial.')
    }

    const [leftStored, rightStored] = await Promise.all([
        dependencies.getStoredIndividualAnalysis(leftId),
        dependencies.getStoredIndividualAnalysis(rightId),
    ])
    const comparison = buildIndividualAnalysisComparison(leftStored.item, rightStored.item)
    if (!comparison.compatible) throw new Error(comparison.reason)

    const dataset = await dependencies.getDatasetById(leftStored.item.dataset_id)
    const image = dataset?.images?.find((candidate) => candidate.id === leftStored.item.image_id)
    if (!image) {
        throw new Error('La imagen original asociada ya no está disponible en el dataset.')
    }

    return {
        status: 'ready',
        left: buildSavedReference(leftStored, dataset, image),
        right: buildSavedReference(rightStored, dataset, image),
        comparison,
        dataset,
        image,
    }
}

export async function loadIndividualAnalysisComparison({
    transientPair,
    leftId,
    rightId,
    getStoredIndividualAnalysis,
    getDatasetById,
}) {
    if (transientPair) return buildTransientComparisonState(transientPair)
    return buildStoredComparisonState(leftId, rightId, {
        getStoredIndividualAnalysis,
        getDatasetById,
    })
}

export function getComparisonLayerCaption(layer, taskType, stage) {
    if (layer === 'perturbation') return 'Mapa amplificado de la perturbación'
    if (layer === 'explanation') {
        return taskType === 'detection' ? 'Eigen-CAM persistido' : 'Grad-CAM persistido'
    }
    return stage === 'original' ? 'Resultado original' : 'Resultado tras el ataque'
}

export function getComparisonExplanationMessage({
    taskType,
    comparisonKind,
    explanationState,
}) {
    const explanationName = taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'
    if (comparisonKind === 'model') {
        return `la capa solo aparece si ambos análisis conservaron su ${explanationName} al ejecutarse. Si falta alguno no se regenera, porque la explicación depende del modelo activo y el historial no cambia ese estado global.`
    }
    if (explanationState.left || explanationState.right) {
        return `solo uno de los análisis conserva ${explanationName}; la capa se oculta para evitar una comparación incompleta.`
    }
    return `ninguno de los dos análisis conserva ${explanationName}; el historial no recalcula explicaciones al abrir una comparación.`
}

export function getComparisonKeyboardAction(key, zoom, minimumZoom) {
    const movements = {
        ArrowLeft: { type: 'pan', x: 24, y: 0 },
        ArrowRight: { type: 'pan', x: -24, y: 0 },
        ArrowUp: { type: 'pan', x: 0, y: 24 },
        ArrowDown: { type: 'pan', x: 0, y: -24 },
    }
    if (movements[key] && zoom > minimumZoom) return movements[key]
    if (key === '+' || key === '=') return { type: 'zoom', direction: 1 }
    if (key === '-') return { type: 'zoom', direction: -1 }
    if (key === '0') return { type: 'reset' }
    return null
}
