export function formatRatio(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—'
    }

    return `${(Number(value) * 100).toFixed(digits)} %`
}

export function formatPoints(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—'
    }

    return `${(Number(value) * 100).toFixed(digits)} pp`
}

export function formatDecimal(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—'
    }

    return Number(value).toFixed(digits)
}

export function formatInteger(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—'
    }

    return new Intl.NumberFormat('es-ES').format(Number(value))
}

export function formatAvailableMetric(metric, formatter = formatRatio) {
    if (!metric?.available || metric.value === null || metric.value === undefined) {
        return '—'
    }

    return formatter(metric.value)
}
