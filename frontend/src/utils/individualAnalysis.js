export const CLASSIFICATION_ANALYSIS_PANELS = Object.freeze([
    'clean',
    'attack',
    'explanation',
])

export const DETECTION_ANALYSIS_PANELS = Object.freeze([
    'clean',
    'attack',
    'explanation',
])

export function normalizeClassificationPanel(panel) {
    return CLASSIFICATION_ANALYSIS_PANELS.includes(panel) ? panel : 'clean'
}

export function normalizeDetectionPanel(panel) {
    return DETECTION_ANALYSIS_PANELS.includes(panel) ? panel : 'clean'
}

export function formatAnalysisProbability(value, digits = 2) {
    const numericValue = Number(value)

    if (!Number.isFinite(numericValue)) {
        return '—'
    }

    return `${(numericValue * 100).toFixed(digits)}%`
}

export function formatSignedPercentagePoints(value, digits = 2) {
    const numericValue = Number(value)

    if (!Number.isFinite(numericValue)) {
        return '—'
    }

    const percentagePoints = numericValue * 100
    const sign = percentagePoints > 0 ? '+' : ''
    return `${sign}${percentagePoints.toFixed(digits)} pp`
}


export function formatSignedInteger(value) {
    const numericValue = Number(value)

    if (!Number.isFinite(numericValue)) {
        return '—'
    }

    const rounded = Math.trunc(numericValue)
    return `${rounded > 0 ? '+' : ''}${rounded}`
}

export function resolveDetectionExplanation({
    attackResult,
    explainabilityResult,
}) {
    if (
        attackResult?.original_explanation_image_url &&
        attackResult?.adversarial_explanation_image_url
    ) {
        return {
            mode: 'comparison',
            original: {
                imageUrl: attackResult.original_explanation_image_url,
                detections: attackResult.original_detections ?? [],
            },
            adversarial: {
                imageUrl: attackResult.adversarial_explanation_image_url,
                detections: attackResult.adversarial_detections ?? [],
            },
        }
    }

    if (explainabilityResult?.explanation_image_url) {
        return {
            mode: 'single',
            original: {
                imageUrl: explainabilityResult.explanation_image_url,
                detections: explainabilityResult.detections ?? [],
            },
            adversarial: null,
        }
    }

    return null
}

export function resolveClassificationExplanation({
    attackResult,
    explainabilityResult,
}) {
    if (
        attackResult?.original_gradcam_image_url &&
        attackResult?.adversarial_gradcam_image_url
    ) {
        return {
            mode: 'comparison',
            original: {
                imageUrl: attackResult.original_gradcam_image_url,
                prediction: attackResult.original_prediction?.prediction ?? null,
                confidence: attackResult.original_prediction?.confidence ?? null,
            },
            adversarial: {
                imageUrl: attackResult.adversarial_gradcam_image_url,
                prediction: attackResult.adversarial_prediction?.prediction ?? null,
                confidence: attackResult.adversarial_prediction?.confidence ?? null,
            },
        }
    }

    if (explainabilityResult?.gradcam_image_url) {
        return {
            mode: 'single',
            original: {
                imageUrl: explainabilityResult.gradcam_image_url,
                prediction: explainabilityResult.prediction ?? null,
                confidence: explainabilityResult.confidence ?? null,
            },
            adversarial: null,
        }
    }

    return null
}

export function analysisImagePosition(selectedIndex, totalImages) {
    const total = Number(totalImages)

    if (!Number.isInteger(selectedIndex) || selectedIndex < 0 || total <= 0) {
        return '— / —'
    }

    return `${selectedIndex + 1} / ${total}`
}

export function resolveAdjacentAnalysisImage(images, selectedIndex, offset, selectionLocked = false) {
    if (selectionLocked || !Number.isInteger(selectedIndex) || selectedIndex < 0) return null
    return images[selectedIndex + offset] ?? null
}
