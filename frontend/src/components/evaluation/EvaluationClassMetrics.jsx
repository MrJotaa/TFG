import { useMemo, useState } from 'react'
import {
    formatAvailableMetric,
    formatInteger,
} from '../../utils/metricFormatting'

const METRIC_OPTIONS = [
    { id: 'f1', label: 'F1' },
    { id: 'recall', label: 'Recall' },
    { id: 'precision', label: 'Precision' },
    { id: 'roc_auc', label: 'ROC-AUC' },
    { id: 'average_precision', label: 'Average Precision' },
]

const SORT_OPTIONS = [
    { id: 'degradation', label: 'Mayor degradación' },
    { id: 'adversarial', label: 'Peor resultado adversarial' },
    { id: 'support', label: 'Mayor soporte' },
    { id: 'name', label: 'Nombre de clase' },
]

function getMetric(classMetric, snapshot, metricId) {
    return classMetric?.[snapshot]?.[metricId] ?? null
}

function getMetricValue(metric) {
    return metric?.available && metric.value !== null && metric.value !== undefined
        ? Number(metric.value)
        : null
}

function getBarWidth(metric) {
    const value = getMetricValue(metric)

    if (value === null) {
        return 0
    }

    return Math.max(0, Math.min(100, value * 100))
}

function getDeltaTone(metric) {
    const value = getMetricValue(metric)

    if (value === null || Math.abs(value) < 0.00005) {
        return 'tone-neutral'
    }

    return value > 0 ? 'tone-danger' : 'tone-success'
}

function formatSignedPoints(metric) {
    const value = getMetricValue(metric)

    if (value === null) {
        return '—'
    }

    const points = value * 100
    const prefix = points > 0 ? '+' : ''

    return `${prefix}${points.toFixed(1)} pp`
}

function compareNullableValues(first, second, direction = 'asc') {
    if (first === null && second === null) {
        return 0
    }

    if (first === null) {
        return 1
    }

    if (second === null) {
        return -1
    }

    return direction === 'desc' ? second - first : first - second
}

function ClassMetricMiniature({ classMetric, metricOption }) {
    const original = getMetric(classMetric, 'original', metricOption.id)
    const adversarial = getMetric(classMetric, 'adversarial', metricOption.id)
    const delta = getMetric(classMetric, 'delta', metricOption.id)
    const tone = getDeltaTone(delta)

    return (
        <div className="class-metric-miniature">
            <span>{metricOption.label}</span>
            <strong>
                {formatAvailableMetric(original)} <small>→</small>{' '}
                {formatAvailableMetric(adversarial)}
            </strong>
            <em className={tone}>{formatSignedPoints(delta)}</em>
        </div>
    )
}

function EvaluationClassMetrics({ evaluation }) {
    const [search, setSearch] = useState('')
    const [selectedMetric, setSelectedMetric] = useState('f1')
    const [sortBy, setSortBy] = useState('degradation')

    const perClass = useMemo(
        () => evaluation?.advanced_metrics?.supervised?.per_class ?? [],
        [evaluation],
    )
    const selectedMetricOption =
        METRIC_OPTIONS.find((metric) => metric.id === selectedMetric) ?? METRIC_OPTIONS[0]

    const filteredMetrics = useMemo(() => {
        const normalizedSearch = search.trim().toLocaleLowerCase('es')

        const matches = perClass.filter((classMetric) => {
            const label = classMetric.class_reference?.label ?? ''
            return label.toLocaleLowerCase('es').includes(normalizedSearch)
        })

        return [...matches].sort((first, second) => {
            if (sortBy === 'name') {
                return (first.class_reference?.label ?? '').localeCompare(
                    second.class_reference?.label ?? '',
                    'es',
                )
            }

            if (sortBy === 'support') {
                return second.support - first.support
            }

            if (sortBy === 'adversarial') {
                return compareNullableValues(
                    getMetricValue(getMetric(first, 'adversarial', selectedMetric)),
                    getMetricValue(getMetric(second, 'adversarial', selectedMetric)),
                )
            }

            return compareNullableValues(
                getMetricValue(getMetric(first, 'delta', selectedMetric)),
                getMetricValue(getMetric(second, 'delta', selectedMetric)),
                'desc',
            )
        })
    }, [perClass, search, selectedMetric, sortBy])

    const totalSupport = perClass.reduce(
        (total, classMetric) => total + Number(classMetric.support ?? 0),
        0,
    )
    const lowSupportCount = perClass.filter((classMetric) => classMetric.support < 2).length

    if (!evaluation?.advanced_metrics?.supervised_status?.available) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Métricas supervisadas</span>
                <h3>No hay métricas por clase disponibles</h3>
                <p>
                    {evaluation?.advanced_metrics?.supervised_status?.reason ??
                        'No hay etiquetas resueltas para calcular esta sección.'}
                </p>
            </div>
        )
    }

    if (!perClass.length) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Métricas supervisadas</span>
                <h3>No se encontraron clases evaluadas</h3>
                <p>La evaluación no contiene resultados desglosados por clase.</p>
            </div>
        )
    }

    return (
        <div className="evaluation-tab-content class-metrics-tab">
            <section className="evaluation-surface class-metrics-header">
                <div>
                    <span className="evaluation-eyebrow">Diagnóstico por clase</span>
                    <h3>Localiza qué categorías son más vulnerables</h3>
                    <p>
                        Compara el rendimiento original y adversarial. Los deltas positivos
                        representan degradación porque se calculan como original menos
                        adversarial.
                    </p>
                </div>

                <div className="class-metrics-summary" aria-label="Resumen por clase">
                    <div>
                        <strong>{formatInteger(perClass.length)}</strong>
                        <span>Clases</span>
                    </div>
                    <div>
                        <strong>{formatInteger(totalSupport)}</strong>
                        <span>Muestras</span>
                    </div>
                    <div className={lowSupportCount ? 'warning' : ''}>
                        <strong>{formatInteger(lowSupportCount)}</strong>
                        <span>Soporte bajo</span>
                    </div>
                </div>
            </section>

            <section className="class-metrics-toolbar" aria-label="Filtros por clase">
                <label>
                    <span>Buscar clase</span>
                    <input
                        type="search"
                        value={search}
                        onChange={(event) => setSearch(event.target.value)}
                        placeholder="Ej. sports car"
                    />
                </label>

                <label>
                    <span>Métrica principal</span>
                    <select
                        value={selectedMetric}
                        onChange={(event) => setSelectedMetric(event.target.value)}
                    >
                        {METRIC_OPTIONS.map((metric) => (
                            <option key={metric.id} value={metric.id}>
                                {metric.label}
                            </option>
                        ))}
                    </select>
                </label>

                <label>
                    <span>Ordenar por</span>
                    <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
                        {SORT_OPTIONS.map((option) => (
                            <option key={option.id} value={option.id}>
                                {option.label}
                            </option>
                        ))}
                    </select>
                </label>
            </section>

            {lowSupportCount > 0 && (
                <div className="evaluation-warning-banner">
                    <strong>Interpretación limitada por soporte bajo.</strong>
                    <span>
                        {formatInteger(lowSupportCount)} clases tienen menos de dos muestras.
                        Sus métricas y curvas pueden variar mucho con una sola imagen.
                    </span>
                </div>
            )}

            {filteredMetrics.length ? (
                <section className="class-metrics-list" aria-label="Métricas por clase">
                    {filteredMetrics.map((classMetric) => {
                        const classReference = classMetric.class_reference
                        const original = getMetric(classMetric, 'original', selectedMetric)
                        const adversarial = getMetric(
                            classMetric,
                            'adversarial',
                            selectedMetric,
                        )
                        const delta = getMetric(classMetric, 'delta', selectedMetric)
                        const warning = original?.warning ?? adversarial?.warning

                        return (
                            <article
                                key={classReference.index}
                                className="class-metric-card"
                            >
                                <header className="class-metric-card-header">
                                    <div className="class-metric-identity">
                                        <span>Clase #{classReference.index}</span>
                                        <h4>{classReference.label}</h4>
                                    </div>

                                    <div className="class-metric-badges">
                                        <span>Soporte {formatInteger(classMetric.support)}</span>
                                        {warning && <strong>{warning}</strong>}
                                    </div>
                                </header>

                                <div className="class-metric-main-comparison">
                                    <div>
                                        <span className="class-metric-main-label">
                                            {selectedMetricOption.label}
                                        </span>

                                        <div className="class-metric-bar-row">
                                            <div className="class-metric-bar-label">
                                                <span>Original</span>
                                                <strong>
                                                    {formatAvailableMetric(original)}
                                                </strong>
                                            </div>
                                            <div className="class-metric-track">
                                                <span
                                                    className="tone-original"
                                                    style={{ width: `${getBarWidth(original)}%` }}
                                                />
                                            </div>
                                        </div>

                                        <div className="class-metric-bar-row">
                                            <div className="class-metric-bar-label">
                                                <span>Adversarial</span>
                                                <strong>
                                                    {formatAvailableMetric(adversarial)}
                                                </strong>
                                            </div>
                                            <div className="class-metric-track">
                                                <span
                                                    className="tone-adversarial"
                                                    style={{
                                                        width: `${getBarWidth(adversarial)}%`,
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <div className={`class-metric-delta ${getDeltaTone(delta)}`}>
                                        <span>Delta</span>
                                        <strong>{formatSignedPoints(delta)}</strong>
                                        <small>original − adversarial</small>
                                    </div>
                                </div>

                                <div className="class-metric-mini-grid">
                                    {METRIC_OPTIONS.map((metricOption) => (
                                        <ClassMetricMiniature
                                            key={metricOption.id}
                                            classMetric={classMetric}
                                            metricOption={metricOption}
                                        />
                                    ))}
                                </div>
                            </article>
                        )
                    })}
                </section>
            ) : (
                <div className="evaluation-empty-panel compact">
                    <h3>No hay coincidencias</h3>
                    <p>Prueba con otro nombre de clase.</p>
                </div>
            )}
        </div>
    )
}

export default EvaluationClassMetrics
