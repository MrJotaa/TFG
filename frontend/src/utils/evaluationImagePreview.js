export const EVALUATION_PREVIEW_MIN_ZOOM = 1
export const EVALUATION_PREVIEW_MAX_ZOOM = 4
export const EVALUATION_PREVIEW_ZOOM_STEP = 0.25

export function clampEvaluationPreviewZoom(value) {
    const numericValue = Number(value)
    if (!Number.isFinite(numericValue)) return EVALUATION_PREVIEW_MIN_ZOOM

    return Math.min(
        EVALUATION_PREVIEW_MAX_ZOOM,
        Math.max(EVALUATION_PREVIEW_MIN_ZOOM, numericValue),
    )
}

export function stepEvaluationPreviewZoom(current, direction) {
    const delta = direction > 0
        ? EVALUATION_PREVIEW_ZOOM_STEP
        : -EVALUATION_PREVIEW_ZOOM_STEP

    return clampEvaluationPreviewZoom(Number(current) + delta)
}

export function getEvaluationPreviewPanBounds({
    viewportWidth,
    viewportHeight,
    imageWidth,
    imageHeight,
    zoom,
}) {
    const safeZoom = clampEvaluationPreviewZoom(zoom)
    const width = Math.max(0, Number(imageWidth) || 0)
    const height = Math.max(0, Number(imageHeight) || 0)
    const viewportW = Math.max(0, Number(viewportWidth) || 0)
    const viewportH = Math.max(0, Number(viewportHeight) || 0)

    return {
        x: Math.max(0, (width * safeZoom - viewportW) / 2),
        y: Math.max(0, (height * safeZoom - viewportH) / 2),
    }
}

export function clampEvaluationPreviewPan(position, bounds) {
    const x = Number(position?.x) || 0
    const y = Number(position?.y) || 0
    const maxX = Math.max(0, Number(bounds?.x) || 0)
    const maxY = Math.max(0, Number(bounds?.y) || 0)

    return {
        x: Math.min(maxX, Math.max(-maxX, x)),
        y: Math.min(maxY, Math.max(-maxY, y)),
    }
}
