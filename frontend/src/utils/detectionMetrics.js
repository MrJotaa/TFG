const UNAVAILABLE_REASON_MESSAGES = {
    no_annotation_snapshot:
        'El conjunto de datos no tiene anotaciones de detección disponibles.',
    no_reviewed_images:
        'Todavía no hay imágenes revisadas en las anotaciones de detección.',
    no_resolved_categories:
        'Ninguna categoría anotada se ha podido resolver contra el modelo.',
    no_resolved_ground_truth_objects:
        'No hay objetos de referencia evaluables con categorías resueltas.',
}

export const COCO_METRIC_ROWS = [
    ['map_50_95', 'mAP@[0.50:0.95]', 'mAP COCO promediado en diez umbrales de IoU; es la referencia principal.'],
    ['ap_50', 'AP@0.50', 'AP con un umbral IoU mínimo de 0,50.'],
    ['ap_75', 'AP@0.75', 'AP con una localización más estricta: IoU mínimo de 0,75.'],
    ['ar_1', 'AR@1', 'AR permitiendo como máximo una detección por imagen.'],
    ['ar_10', 'AR@10', 'AR permitiendo como máximo diez detecciones por imagen.'],
    ['ar_100', 'AR@100', 'AR permitiendo como máximo cien detecciones por imagen.'],
    ['ap_small', 'AP pequeño', 'AP para objetos con área inferior a 32² píxeles.'],
    ['ap_medium', 'AP mediano', 'AP para objetos con área entre 32² y 96² píxeles.'],
    ['ap_large', 'AP grande', 'AP para objetos con área igual o superior a 96² píxeles.'],
    ['ar_small', 'AR pequeño', 'AR para objetos pequeños según la escala COCO.'],
    ['ar_medium', 'AR mediano', 'AR para objetos medianos según la escala COCO.'],
    ['ar_large', 'AR grande', 'AR para objetos grandes según la escala COCO.'],
]

export const TIDE_ERROR_DEFINITIONS = {
    classification: {
        label: 'Clasificación',
        shortLabel: 'Clas.',
        description: 'La caja localiza un objeto, pero predice una clase incorrecta.',
    },
    localization: {
        label: 'Localización',
        shortLabel: 'Localiz.',
        description: 'La clase es correcta, pero la caja no alcanza IoU 0,50.',
    },
    classification_localization: {
        label: 'Clasificación y localización',
        shortLabel: 'Ambos',
        description: 'La detección falla simultáneamente en la clase y en la localización.',
    },
    duplicate: {
        label: 'Duplicados',
        shortLabel: 'Duplic.',
        description: 'Varias detecciones compiten por un objeto ya asignado.',
    },
    background: {
        label: 'Fondo',
        shortLabel: 'Fondo',
        description: 'La detección no solapa suficientemente ningún objeto (IoU inferior a 0,10).',
    },
    missed: {
        label: 'Objetos omitidos',
        shortLabel: 'Omitidos',
        description: 'Un objeto anotado no queda cubierto por ninguna detección válida.',
    },
    false_positive: {
        label: 'Falsos positivos',
        shortLabel: 'FP',
        description: 'Impacto global de corregir todos los falsos positivos sin modificar el recall.',
    },
    false_negative: {
        label: 'Falsos negativos',
        shortLabel: 'FN',
        description: 'Impacto global de recuperar el recall perfecto sin modificar la precisión.',
    },
}

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

export function formatTidePoints(value, digits = 1, signed = false) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return '—'
    }

    const numericValue = Number(value)
    const prefix = signed && numericValue > 0 ? '+' : ''
    return `${prefix}${numericValue.toFixed(digits)} pp AP`
}

export function getTideErrorRows(tideAnalysis) {
    const originalByType = new Map(
        (tideAnalysis?.original?.main_errors ?? []).map((item) => [item.error_type, item]),
    )
    const adversarialByType = new Map(
        (tideAnalysis?.adversarial?.main_errors ?? []).map((item) => [item.error_type, item]),
    )
    const changeByType = new Map(
        (tideAnalysis?.changes ?? []).map((item) => [item.error_type, item]),
    )

    return Object.keys(TIDE_ERROR_DEFINITIONS)
        .filter((errorType) => originalByType.has(errorType))
        .map((errorType) => ({
            errorType,
            ...TIDE_ERROR_DEFINITIONS[errorType],
            original: originalByType.get(errorType),
            adversarial: adversarialByType.get(errorType),
            change: changeByType.get(errorType),
        }))
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
