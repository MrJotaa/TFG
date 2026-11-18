export const ANNOTATION_STATUS_LABELS = {
    unreviewed: 'Sin revisar',
    reviewed_empty: 'Revisada sin objetos',
    reviewed_with_objects: 'Revisada con objetos',
}

export function getAnnotationStatusLabel(status) {
    return ANNOTATION_STATUS_LABELS[status] || ANNOTATION_STATUS_LABELS.unreviewed
}

export function shouldConfirmCocoReplacement(coverage) {
    return Boolean(coverage?.has_annotation_snapshot)
}

export function canSubmitCocoImport({ file, importing }) {
    return Boolean(file) && !importing
}

export function getCoverageAfterCocoImport(result) {
    return result.coverage
}

export function formatCocoImportError(error) {
    const details = error?.details
    if (!details) {
        return error?.message || 'No se pudieron importar las anotaciones.'
    }

    const context = [
        details.section ? `sección: ${details.section}` : null,
        details.field ? `campo: ${details.field}` : null,
        details.coco_id !== undefined ? `ID COCO: ${details.coco_id}` : null,
        details.image_name ? `imagen: ${details.image_name}` : null,
    ].filter(Boolean)

    return context.length > 0
        ? `${details.message} (${context.join(', ')}). No se aplicó ningún cambio.`
        : `${details.message}. No se aplicó ningún cambio.`
}
