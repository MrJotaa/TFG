/**
 * Presentación del criterio de éxito de detección declarado por el backend.
 * Nunca se recalcula attack_success en el frontend: los resultados históricos
 * sin metadatos conservan el significado de la versión 1.0.
 */
export const MATCHED_DETECTION_SUCCESS = 'matched_detections_iou50_v1'
export const LEGACY_DETECTION_SUCCESS = 'legacy_count_or_confidence_v1'

export function getDetectionSuccessPresentation(result) {
    const criterion = result?.attack_success_criterion
        ?? result?.summary?.attack_success_criterion
    const version = result?.metrics_schema_version
        ?? result?.summary?.metrics_schema_version
    const matched = criterion === MATCHED_DETECTION_SUCCESS
        || (criterion == null && version === '1.1')

    if (matched) {
        return {
            criterion: MATCHED_DETECTION_SUCCESS,
            version: '1.1',
            rateLabel: 'Tasa de alteración de detecciones',
            rateHelper: 'Al menos una detección original pierde su pareja de la misma clase con IoU ≥ 0,50.',
            positiveLabel: 'Alteración de detecciones',
            negativeLabel: 'Correspondencias conservadas',
            description: 'Criterio v1.1: correspondencia uno a uno por clase e IoU ≥ 0,50. Una bajada aislada de confianza no cuenta como éxito. Se compara con las predicciones originales, no con las anotaciones reales.',
            historyLabel: 'Criterio v1.1 · correspondencia de detecciones',
        }
    }

    return {
        criterion: LEGACY_DETECTION_SUCCESS,
        version: '1.0',
        rateLabel: 'Tasa de cambio (criterio histórico)',
        rateHelper: 'Menos detecciones o una confianza media inferior; incluye reducciones pequeñas.',
        positiveLabel: 'Cambio según criterio histórico',
        negativeLabel: 'Criterio histórico no cumplido',
        description: 'Criterio histórico v1.0: disminución del número de detecciones o de la confianza media, aunque sea mínima. No equivale necesariamente a un fallo adversario.',
        historyLabel: 'Criterio v1.0 · conteo o confianza',
    }
}

export function detectionSuccessLabel(result, success) {
    const copy = getDetectionSuccessPresentation(result)
    if (success === true) return copy.positiveLabel
    if (success === false) return copy.negativeLabel
    return 'Resultado sin indicador de éxito'
}

/**
 * Explicación observable del resultado de correspondencia.
 * No vuelve a emparejar cajas ni modifica attack_success: sólo interpreta los
 * campos calculados por el backend. También se usa en resultados históricos.
 */
export function detectionOutcomeExplanation(result, metrics = result?.comparison ?? result) {
    const copy = getDetectionSuccessPresentation(result)
    if (copy.version !== '1.1' || typeof metrics?.attack_success !== 'boolean') {
        return copy.description
    }

    const originals = Number(metrics.original_detection_count)
    const following = Number(metrics.adversarial_detection_count)
    const validOriginals = Number.isInteger(originals) && originals >= 0
    const validFollowing = Number.isInteger(following) && following >= 0
    let message

    if (validOriginals && originals === 0) {
        message = 'No había detecciones originales que comparar; no puede medirse una pérdida de correspondencia.'
    } else if (metrics.attack_success) {
        message = 'Al menos una detección original ha perdido su pareja de la misma clase con IoU ≥ 0,50. Puede deberse a desaparición, cambio de clase o desplazamiento de su caja.'
    } else {
        const count = validOriginals
            ? `Las ${originals} detecciones originales conservan`
            : 'Las detecciones originales conservan'
        message = `${count} una pareja de su clase con IoU ≥ 0,50. Eso no significa que las cajas sean idénticas: pueden haber cambiado visiblemente sin superar el umbral de alteración.`
    }

    if (validOriginals && validFollowing && originals !== following) {
        const difference = Math.abs(following - originals)
        message += following > originals
            ? ` Aparecen ${difference} detecciones adicionales.`
            : ` Hay ${difference} detecciones menos tras el ataque.`
    }

    const clean = metrics.average_original_confidence
    const adversarial = metrics.average_adversarial_confidence
    if (typeof clean === 'number' && typeof adversarial === 'number'
        && Number.isFinite(clean) && Number.isFinite(adversarial)) {
        const difference = (clean - adversarial) * 100
        if (difference > 0) {
            message += ` La confianza media bajó ${difference.toFixed(2)} puntos porcentuales; por sí sola, esa caída no determina el resultado.`
        } else if (difference < 0) {
            message += ` La confianza media subió ${(-difference).toFixed(2)} puntos porcentuales.`
        }
    }
    return message
}

const SKIP_STATUSES = {
    skipped_no_references: 'Square no se ejecutó: no había detecciones originales fiables.',
    skipped_zero_epsilon: 'Square no se ejecutó: no se permitía modificar la imagen.',
    skipped_insufficient_budget: 'Square no se ejecutó: consultas insuficientes.',
    skipped_unsupported_geometry: 'Square no se ejecutó: la imagen tiene una geometría incompatible.',
}

/** Objetivo particular de Square, sin confundirlo con la métrica común. */
export function squareObjectivePresentation(trace) {
    if (!trace) return null
    const skipped = SKIP_STATUSES[trace.status]
    const hasScores = typeof trace.clean_score === 'number'
        && typeof trace.best_score === 'number'
        && Number.isFinite(trace.clean_score) && Number.isFinite(trace.best_score)
    const scores = hasScores
        ? `Conservación de detecciones: ${trace.clean_score.toFixed(3)} → ${trace.best_score.toFixed(3)}`
        : null
    const reduction = hasScores && trace.clean_score > 0
        ? (1 - trace.best_score / trace.clean_score) * 100
        : null
    const reductionText = reduction !== null && reduction > 0
        ? ` (${reduction.toFixed(1)} % menos)`
        : ''
    const queries = Number.isInteger(trace.queries) && Number.isInteger(trace.max_queries)
        ? `Consultas al detector: ${trace.queries} de ${trace.max_queries}`
        : null

    return {
        label: skipped ?? (trace.objective_reached
            ? 'Square alcanzó su objetivo de debilitamiento'
            : 'Square no alcanzó su objetivo de debilitamiento'),
        detail: [scores ? scores + reductionText : null, queries].filter(Boolean).join(' · '),
        note: skipped
            ? 'La imagen puede conservarse sin modificaciones.'
            : 'Cuanto menor sea la puntuación, menos se conservan la confianza o el solapamiento de las detecciones originales. Square busca reducirla a 0,80 o menos; esto no implica necesariamente perder una correspondencia de clase y caja (IoU ≥ 0,50).',
    }
}
