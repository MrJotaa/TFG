const UNAVAILABLE_REASON_MESSAGES = {
    no_annotation_snapshot:
        'El dataset no tiene un snapshot de anotaciones disponible.',
    no_reviewed_images:
        'Todavía no hay imágenes revisadas en el snapshot de anotaciones.',
    no_resolved_categories:
        'Ninguna categoría anotada se ha podido resolver contra el modelo.',
    no_resolved_ground_truth_objects:
        'No hay objetos ground truth evaluables con categorías resueltas.',
}

export const COCO_METRIC_ROWS = [
    ['map_50_95', 'mAP@[.50:.95]'],
    ['ap_50', 'AP@0.50'],
    ['ap_75', 'AP@0.75'],
    ['ar_1', 'AR@1'],
    ['ar_10', 'AR@10'],
    ['ar_100', 'AR@100'],
    ['ap_small', 'AP small'],
    ['ap_medium', 'AP medium'],
    ['ap_large', 'AP large'],
    ['ar_small', 'AR small'],
    ['ar_medium', 'AR medium'],
    ['ar_large', 'AR large'],
]

export const OBJECT_OUTCOMES = [
    ['preserved_count', 'Preservados', 'preserved'],
    ['suppressed_count', 'Suprimidos', 'suppressed'],
    ['recovered_count', 'Recuperados', 'recovered'],
    ['missed_both_count', 'No detectados en ambas', 'missed'],
]

export function getDetectionUnavailableMessage(reason) {
    return (
        UNAVAILABLE_REASON_MESSAGES[reason] ??
        'Las métricas supervisadas no están disponibles para esta evaluación.'
    )
}

export function formatSignedPercentagePoints(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return '—'
    }

    const numericValue = Number(value) * 100
    const prefix = numericValue > 0 ? '+' : ''
    return `${prefix}${numericValue.toFixed(digits)} pp`
}

export function formatSignedPercent(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return '—'
    }

    const numericValue = Number(value) * 100
    const prefix = numericValue > 0 ? '+' : ''
    return `${prefix}${numericValue.toFixed(digits)} %`
}

export function getDeltaTone(value) {
    if (value === null || value === undefined || Math.abs(Number(value)) < 0.0000005) {
        return 'neutral'
    }

    return Number(value) > 0 ? 'danger' : 'success'
}

export function getObjectOutcomeItems(aggregate) {
    return OBJECT_OUTCOMES.map(([field, label, tone]) => ({
        field,
        label,
        tone,
        count: Number(aggregate?.[field] ?? 0),
    }))
}

export function mergePrecisionRecallCurves(originalCurve, adversarialCurve) {
    const points = new Map()

    for (const point of originalCurve?.points ?? []) {
        points.set(Number(point.recall), {
            recall: Number(point.recall),
            original: Number(point.precision),
        })
    }

    for (const point of adversarialCurve?.points ?? []) {
        const recall = Number(point.recall)
        points.set(recall, {
            ...points.get(recall),
            recall,
            adversarial: Number(point.precision),
        })
    }

    return [...points.values()].sort((left, right) => left.recall - right.recall)
}

export function findClassMetrics(analysis, modelClassId) {
    return analysis?.per_class?.find(
        (entry) => String(entry.reference?.model_class_id) === String(modelClassId),
    )
}

export function formatBoundingBox(box) {
    if (!box) {
        return '—'
    }

    return [box.x1, box.y1, box.x2, box.y2]
        .map((value) => Number(value).toFixed(1))
        .join(', ')
}
