export const RESULT_RETENTION_OPTIONS = [
    { value: 3600, label: '1 hora' },
    { value: 7200, label: '2 horas' },
    { value: 86400, label: '24 horas' },
    { value: 172800, label: '2 días' },
    { value: 604800, label: '7 días' },
]

const RELATIVE_TIME = new Intl.RelativeTimeFormat('es', { numeric: 'auto' })

export function formatDateTime(value) {
    if (!value) return '—'
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return '—'
    return new Intl.DateTimeFormat('es', {
        dateStyle: 'medium',
        timeStyle: 'short',
    }).format(date)
}

export function formatRemainingTime(expiresAt, now = Date.now()) {
    if (!expiresAt) return 'Sin fecha todavía'
    const remainingSeconds = Math.ceil((new Date(expiresAt).getTime() - now) / 1000)
    if (remainingSeconds <= 0) return 'Caducado'
    if (remainingSeconds < 60) return `${remainingSeconds} s`
    const remainingMinutes = Math.ceil(remainingSeconds / 60)
    if (remainingMinutes < 60) return `${remainingMinutes} min`
    const remainingHours = Math.ceil(remainingMinutes / 60)
    if (remainingHours < 48) return `${remainingHours} h`
    return `${Math.ceil(remainingHours / 24)} días`
}


export function evaluationExpirationLabel(item, now = Date.now()) {
    if (item?.status === 'running') return 'Al finalizar'
    if (item?.preserved) return 'Indefinida'
    return formatRemainingTime(item?.expires_at, now)
}

export function historyTotalLabel(kind, total) {
    if (kind === 'evaluations') {
        const noun = total === 1 ? 'evaluación vigente' : 'evaluaciones vigentes'
        return `${total} ${noun}`
    }

    const noun = total === 1 ? 'análisis guardado' : 'análisis guardados'
    return `${total} ${noun}`
}

export function formatCreatedAgo(createdAt, now = Date.now()) {
    const elapsedSeconds = Math.round((new Date(createdAt).getTime() - now) / 1000)
    if (!Number.isFinite(elapsedSeconds)) return 'fecha desconocida'
    if (Math.abs(elapsedSeconds) < 60) return RELATIVE_TIME.format(elapsedSeconds, 'second')
    const elapsedMinutes = Math.round(elapsedSeconds / 60)
    if (Math.abs(elapsedMinutes) < 60) return RELATIVE_TIME.format(elapsedMinutes, 'minute')
    const elapsedHours = Math.round(elapsedMinutes / 60)
    if (Math.abs(elapsedHours) < 48) return RELATIVE_TIME.format(elapsedHours, 'hour')
    return RELATIVE_TIME.format(Math.round(elapsedHours / 24), 'day')
}

export function isNearExpiration(item, now = Date.now()) {
    if (!item?.expires_at || !item?.retention_seconds) return false
    const remaining = new Date(item.expires_at).getTime() - now
    const threshold = Math.min(item.retention_seconds * 100, 2 * 60 * 60 * 1000)
    return remaining > 0 && remaining <= threshold
}

export function historySummary(item) {
    const summary = item?.summary
    if (!summary) return item?.status === 'running' ? 'Calculando…' : 'Sin resumen'
    if (item.task_type === 'classification') {
        if (summary.original_accuracy != null && summary.adversarial_accuracy != null) {
            return `Accuracy ${(summary.original_accuracy * 100).toFixed(1)}% → ${(summary.adversarial_accuracy * 100).toFixed(1)}%`
        }
        return `${summary.changed_predictions ?? 0} predicciones cambiadas`
    }
    return `${summary.total_original_detections ?? 0} → ${summary.total_adversarial_detections ?? 0} detecciones`
}
