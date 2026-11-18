export function evaluationProgressPercent(progress) {
    if (!progress || !Number.isFinite(progress.total) || progress.total <= 0) {
        return 0
    }
    const processed = Math.min(
        Math.max(Number(progress.processed) || 0, 0),
        progress.total,
    )
    return Math.round((processed / progress.total) * 100)
}

export function evaluationProgressMessage(progress) {
    if (!progress) {
        return ''
    }
    if (progress.phase === 'metrics') {
        return `${progress.total} de ${progress.total} imágenes procesadas. Calculando métricas finales.`
    }
    return `${progress.processed} de ${progress.total} imágenes procesadas.`
}
