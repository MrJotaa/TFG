export const DATASET_TASK_LABELS = {
    classification: 'Clasificación',
    detection: 'Detección',
}

export function getDatasetTaskLabel(taskType) {
    return DATASET_TASK_LABELS[taskType] || 'Tipo sin definir'
}

export function datasetSupportsAnnotations(dataset) {
    return dataset?.task_type === 'detection'
}

export function getClassificationLabelStats(images = []) {
    const total = images.length
    const labeled = images.filter((image) => Boolean(image.label?.trim())).length

    return {
        total,
        labeled,
        unlabeled: Math.max(total - labeled, 0),
        percentage: total === 0 ? 0 : Math.round((labeled / total) * 100),
    }
}

export function filterClassificationImages(images = [], filter = 'all') {
    if (filter === 'labeled') {
        return images.filter((image) => Boolean(image.label?.trim()))
    }
    if (filter === 'unlabeled') {
        return images.filter((image) => !image.label?.trim())
    }
    return images
}

export function getSelectedFilesSummary(files = []) {
    const normalizedFiles = Array.from(files)
    const totalBytes = normalizedFiles.reduce(
        (sum, file) => sum + (Number(file.size) || 0),
        0,
    )

    return {
        count: normalizedFiles.length,
        totalBytes,
        totalMegabytes: totalBytes / (1024 * 1024),
        previewNames: normalizedFiles.slice(0, 3).map((file) => file.name),
        remainingCount: Math.max(normalizedFiles.length - 3, 0),
    }
}

export function getDatasetSidebarMeta(dataset) {
    const type = getDatasetTaskLabel(dataset?.task_type)
    const count = dataset?.image_count ?? 0
    return `${type} · ${count} ${count === 1 ? 'imagen' : 'imágenes'}`
}
