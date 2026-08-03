export const EXPLORER_MIN_ZOOM = 1
export const EXPLORER_MAX_ZOOM = 4
export const EXPLORER_ZOOM_STEP = 0.25

export function clampExplorerZoom(value) {
    const numericValue = Number(value)

    if (!Number.isFinite(numericValue)) {
        return EXPLORER_MIN_ZOOM
    }

    return Math.min(
        EXPLORER_MAX_ZOOM,
        Math.max(EXPLORER_MIN_ZOOM, numericValue),
    )
}

export function resolveCleanDetectionResult({ detectionResult, attackResult }) {
    if (detectionResult) {
        return detectionResult
    }

    if (!attackResult) {
        return null
    }

    return {
        annotated_image_url: attackResult.original_annotated_image_url,
        detections: attackResult.original_detections,
    }
}

export function buildExplorerSources({ taskType, selectedImage, attackResult }) {
    if (!attackResult) {
        return null
    }

    const originalSourceUrl = selectedImage?.url ?? null

    if (taskType === 'detection') {
        return {
            taskType,
            originalSourceUrl,
            adversarialSourceUrl: attackResult.adversarial_image_url,
            originalAnnotatedUrl: attackResult.original_annotated_image_url,
            adversarialAnnotatedUrl:
                attackResult.adversarial_annotated_image_url,
            originalExplanationUrl:
                attackResult.original_explanation_image_url,
            adversarialExplanationUrl:
                attackResult.adversarial_explanation_image_url,
            perturbationUrl: attackResult.perturbation_image_url,
            supportsAnnotations: true,
            supportsExplanations: Boolean(
                attackResult.original_explanation_image_url &&
                    attackResult.adversarial_explanation_image_url,
            ),
        }
    }

    return {
        taskType: 'classification',
        originalSourceUrl,
        adversarialSourceUrl: attackResult.adversarial_image_url,
        originalAnnotatedUrl: null,
        adversarialAnnotatedUrl: null,
        originalExplanationUrl: attackResult.original_gradcam_image_url,
        adversarialExplanationUrl: attackResult.adversarial_gradcam_image_url,
        perturbationUrl: attackResult.perturbation_image_url,
        supportsAnnotations: false,
        supportsExplanations: Boolean(
            attackResult.original_gradcam_image_url &&
                attackResult.adversarial_gradcam_image_url,
        ),
    }
}

export function resolveExplorerPair({ sources, layer, showAnnotations }) {
    if (!sources) {
        return { original: null, adversarial: null }
    }

    if (layer === 'explanation' && sources.supportsExplanations) {
        return {
            original: sources.originalExplanationUrl,
            adversarial: sources.adversarialExplanationUrl,
        }
    }

    if (showAnnotations && sources.supportsAnnotations) {
        return {
            original:
                sources.originalAnnotatedUrl ?? sources.originalSourceUrl,
            adversarial:
                sources.adversarialAnnotatedUrl ?? sources.adversarialSourceUrl,
        }
    }

    return {
        original: sources.originalSourceUrl,
        adversarial: sources.adversarialSourceUrl,
    }
}
