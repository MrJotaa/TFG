import { useId, useMemo, useState } from 'react'
import {
    formatAvailableMetric,
    formatPoints,
    formatRatio,
} from '../../utils/metricFormatting'

const CURVE_TYPES = {
    roc: {
        label: 'ROC',
        title: 'Receiver Operating Characteristic',
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        scoreLabel: 'ROC-AUC',
        description:
            'Muestra la relación entre sensibilidad y falsos positivos al variar el umbral.',
    },
    precision_recall: {
        label: 'Precision-Recall',
        title: 'Precision frente a Recall',
        xLabel: 'Recall',
        yLabel: 'Precision',
        scoreLabel: 'Average Precision',
        description:
            'Resume el equilibrio entre precision y recall para diferentes umbrales.',
    },
}

const SVG_WIDTH = 760
const SVG_HEIGHT = 420
const PADDING = {
    top: 24,
    right: 26,
    bottom: 62,
    left: 68,
}
const TICKS = [0, 0.25, 0.5, 0.75, 1]

function isAvailable(series) {
    return Boolean(series?.available && series.points?.length)
}

function formatSignedPoints(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—'
    }

    const prefix = value > 0 ? '+' : ''
    return `${prefix}${formatPoints(value)}`
}

function getDeltaTone(value) {
    if (value === null || value === undefined || Math.abs(value) < 0.00005) {
        return 'neutral'
    }

    return value > 0 ? 'danger' : 'success'
}

function getScopeData(analysis, scope) {
    if (!analysis) {
        return null
    }

    if (scope === 'micro') {
        return {
            label: 'Micro-average',
            support: null,
            curves: analysis.micro,
        }
    }

    return analysis.per_class?.find(
        (entry) => String(entry.class_reference?.index) === scope,
    )
}

function buildPolyline(points) {
    if (!points?.length) {
        return ''
    }

    const chartWidth = SVG_WIDTH - PADDING.left - PADDING.right
    const chartHeight = SVG_HEIGHT - PADDING.top - PADDING.bottom

    return points
        .map((point) => {
            const x = PADDING.left + Number(point.x) * chartWidth
            const y = PADDING.top + (1 - Number(point.y)) * chartHeight
            return `${x},${y}`
        })
        .join(' ')
}

function CurveChart({ curveType, original, adversarial, scopeLabel }) {
    const titleId = useId()
    const descriptionId = useId()
    const config = CURVE_TYPES[curveType]
    const chartWidth = SVG_WIDTH - PADDING.left - PADDING.right
    const chartHeight = SVG_HEIGHT - PADDING.top - PADDING.bottom
    const hasOriginal = isAvailable(original)
    const hasAdversarial = isAvailable(adversarial)

    if (!hasOriginal && !hasAdversarial) {
        return (
            <div className="curve-unavailable-panel">
                <span className="evaluation-eyebrow">Curva no disponible</span>
                <h4>{scopeLabel}</h4>
                <p>
                    {original?.reason ??
                        adversarial?.reason ??
                        'No hay suficientes datos para representar esta curva.'}
                </p>
            </div>
        )
    }

    return (
        <div className="curve-chart-wrap">
            <svg
                className="curve-chart"
                viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
                role="img"
                aria-labelledby={`${titleId} ${descriptionId}`}
            >
                <title id={titleId}>
                    {config.label} original y adversarial para {scopeLabel}
                </title>
                <desc id={descriptionId}>
                    Comparación de la curva original y adversarial. El eje horizontal es
                    {` ${config.xLabel}`} y el vertical es {config.yLabel}.
                </desc>

                {TICKS.map((tick) => {
                    const x = PADDING.left + tick * chartWidth
                    const y = PADDING.top + (1 - tick) * chartHeight

                    return (
                        <g key={tick}>
                            <line
                                className="curve-grid-line"
                                x1={PADDING.left}
                                y1={y}
                                x2={SVG_WIDTH - PADDING.right}
                                y2={y}
                            />
                            <line
                                className="curve-grid-line"
                                x1={x}
                                y1={PADDING.top}
                                x2={x}
                                y2={SVG_HEIGHT - PADDING.bottom}
                            />
                            <text
                                className="curve-axis-tick"
                                x={PADDING.left - 14}
                                y={y + 4}
                                textAnchor="end"
                            >
                                {tick.toFixed(2)}
                            </text>
                            <text
                                className="curve-axis-tick"
                                x={x}
                                y={SVG_HEIGHT - PADDING.bottom + 24}
                                textAnchor="middle"
                            >
                                {tick.toFixed(2)}
                            </text>
                        </g>
                    )
                })}

                <line
                    className="curve-axis-line"
                    x1={PADDING.left}
                    y1={SVG_HEIGHT - PADDING.bottom}
                    x2={SVG_WIDTH - PADDING.right}
                    y2={SVG_HEIGHT - PADDING.bottom}
                />
                <line
                    className="curve-axis-line"
                    x1={PADDING.left}
                    y1={PADDING.top}
                    x2={PADDING.left}
                    y2={SVG_HEIGHT - PADDING.bottom}
                />

                {curveType === 'roc' && (
                    <line
                        className="curve-reference-line"
                        x1={PADDING.left}
                        y1={SVG_HEIGHT - PADDING.bottom}
                        x2={SVG_WIDTH - PADDING.right}
                        y2={PADDING.top}
                    />
                )}

                {hasOriginal && (
                    <polyline
                        className="curve-series original"
                        points={buildPolyline(original.points)}
                    />
                )}

                {hasAdversarial && (
                    <polyline
                        className="curve-series adversarial"
                        points={buildPolyline(adversarial.points)}
                    />
                )}

                <text
                    className="curve-axis-label"
                    x={PADDING.left + chartWidth / 2}
                    y={SVG_HEIGHT - 14}
                    textAnchor="middle"
                >
                    {config.xLabel}
                </text>
                <text
                    className="curve-axis-label"
                    x={18}
                    y={PADDING.top + chartHeight / 2}
                    textAnchor="middle"
                    transform={`rotate(-90 18 ${PADDING.top + chartHeight / 2})`}
                >
                    {config.yLabel}
                </text>
            </svg>
        </div>
    )
}

function CurveScoreCard({ label, series, tone }) {
    return (
        <article className={`curve-score-card ${tone}`}>
            <span>{label}</span>
            <strong>{series?.available ? formatRatio(series.score) : '—'}</strong>
            <small>{series?.warning ?? series?.reason ?? 'Valor calculado'}</small>
        </article>
    )
}

function EvaluationCurves({ evaluation }) {
    const [curveType, setCurveType] = useState('roc')
    const [scope, setScope] = useState('micro')

    const supervised = evaluation?.advanced_metrics?.supervised
    const analysis = supervised?.[curveType]
    const config = CURVE_TYPES[curveType]

    const classOptions = useMemo(
        () =>
            (analysis?.per_class ?? []).map((entry) => ({
                index: String(entry.class_reference?.index),
                label: entry.class_reference?.label ?? 'Clase sin nombre',
                support: entry.support,
            })),
        [analysis],
    )

    const selectedScope = getScopeData(analysis, scope)
    const original = selectedScope?.curves?.original
    const adversarial = selectedScope?.curves?.adversarial
    const originalScore = original?.available ? Number(original.score) : null
    const adversarialScore = adversarial?.available ? Number(adversarial.score) : null
    const delta =
        originalScore !== null && adversarialScore !== null
            ? originalScore - adversarialScore
            : null

    if (!evaluation?.advanced_metrics?.supervised_status?.available || !supervised) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Análisis por umbral</span>
                <h3>Las curvas no están disponibles</h3>
                <p>
                    {evaluation?.advanced_metrics?.supervised_status?.reason ??
                        'No hay etiquetas resueltas para construir las curvas.'}
                </p>
            </div>
        )
    }

    return (
        <div className="evaluation-tab-content evaluation-curves-tab">
            <section className="evaluation-surface curve-header-card">
                <div>
                    <span className="evaluation-eyebrow">Análisis por umbral</span>
                    <h3>Curvas ROC y Precision-Recall</h3>
                    <p>
                        Superpone el comportamiento original y adversarial y explora el
                        promedio micro o una clase concreta.
                    </p>
                </div>

                <div className="curve-controls">
                    <div
                        className="evaluation-segmented-control"
                        role="group"
                        aria-label="Tipo de curva"
                    >
                        {Object.entries(CURVE_TYPES).map(([id, option]) => (
                            <button
                                key={id}
                                type="button"
                                className={curveType === id ? 'active' : ''}
                                aria-pressed={curveType === id}
                                onClick={() => {
                                    setCurveType(id)
                                    setScope('micro')
                                }}
                            >
                                {option.label}
                            </button>
                        ))}
                    </div>

                    <label>
                        <span>Serie</span>
                        <select
                            value={scope}
                            onChange={(event) => setScope(event.target.value)}
                        >
                            <option value="micro">Micro-average</option>
                            {classOptions.map((option) => (
                                <option key={option.index} value={option.index}>
                                    {option.label} · n={option.support}
                                </option>
                            ))}
                        </select>
                    </label>
                </div>
            </section>

            <section className="curve-score-grid" aria-label="Resumen de curvas">
                <CurveScoreCard label={`${config.scoreLabel} original`} series={original} tone="original" />
                <CurveScoreCard
                    label={`${config.scoreLabel} adversarial`}
                    series={adversarial}
                    tone="adversarial"
                />
                <article className={`curve-score-card ${getDeltaTone(delta)}`}>
                    <span>Degradación</span>
                    <strong>{formatSignedPoints(delta)}</strong>
                    <small>Original menos adversarial</small>
                </article>
                <article className="curve-score-card neutral">
                    <span>Macro original / adversarial</span>
                    <strong>
                        {formatAvailableMetric(analysis?.macro_original)} /{' '}
                        {formatAvailableMetric(analysis?.macro_adversarial)}
                    </strong>
                    <small>Promedio no ponderado entre clases</small>
                </article>
            </section>

            <section className="evaluation-surface curve-main-card">
                <div className="evaluation-section-heading curve-chart-heading">
                    <div>
                        <span className="evaluation-eyebrow">{config.label}</span>
                        <h3>{selectedScope?.class_reference?.label ?? 'Micro-average'}</h3>
                        <p>{config.description}</p>
                    </div>

                    <div className="evaluation-legend-pills" aria-label="Leyenda">
                        <span className="original">Original</span>
                        <span className="adversarial">Adversarial</span>
                        {curveType === 'roc' && <span className="reference">Azar</span>}
                    </div>
                </div>

                <CurveChart
                    curveType={curveType}
                    original={original}
                    adversarial={adversarial}
                    scopeLabel={selectedScope?.class_reference?.label ?? 'Micro-average'}
                />
            </section>

            {(original?.warning || adversarial?.warning) && (
                <div className="evaluation-warning-banner" role="note">
                    <strong>Interpretación limitada por el soporte.</strong>
                    <span>{original?.warning ?? adversarial?.warning}</span>
                </div>
            )}
        </div>
    )
}

export default EvaluationCurves
