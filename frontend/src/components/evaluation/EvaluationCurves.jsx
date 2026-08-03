import { useEffect, useMemo, useRef, useState } from 'react'
import {
    CartesianGrid,
    Line,
    LineChart,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts'
import {
    formatAvailableMetric,
    formatPoints,
    formatRatio,
} from '../../utils/metricFormatting'

const CURVE_TYPES = {
    roc: {
        label: 'ROC',
        xLabel: 'Tasa de falsos positivos',
        yLabel: 'Tasa de verdaderos positivos',
        tooltipXLabel: 'Falsos positivos',
        tooltipYLabel: 'Verdaderos positivos',
        scoreLabel: 'ROC-AUC',
        description:
            'Muestra la relación entre sensibilidad y falsos positivos al variar el umbral.',
    },
    precision_recall: {
        label: 'Precision-Recall',
        xLabel: 'Recall',
        yLabel: 'Precisión',
        tooltipXLabel: 'Recall',
        tooltipYLabel: 'Precisión',
        scoreLabel: 'Precisión media (AP)',
        description:
            'Resume el equilibrio entre precisión y recall para diferentes umbrales.',
    },
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

function formatThreshold(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return '—'
    }

    return new Intl.NumberFormat('es-ES', {
        maximumFractionDigits: 4,
    }).format(Number(value))
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
            label: 'Micro global',
            support: null,
            curves: analysis.micro,
        }
    }

    return analysis.per_class?.find(
        (entry) => String(entry.class_reference?.index) === scope,
    )
}

function normalizeCurvePoints(points) {
    return (points ?? [])
        .map((point) => ({
            x: Number(point.x),
            y: Number(point.y),
            threshold: point.threshold,
        }))
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
}

function CurveTooltip({ active, payload, curveType }) {
    if (!active || !payload?.length) {
        return null
    }

    const config = CURVE_TYPES[curveType]
    const entries = payload.filter((entry) => entry?.payload)

    if (!entries.length) {
        return null
    }

    return (
        <div className="curve-tooltip">
            {entries.map((entry, index) => {
                const point = entry.payload

                return (
                    <div
                        className="curve-tooltip-series"
                        key={`${entry.name}-${point.x}-${index}`}
                    >
                        <strong>
                            <span
                                aria-hidden="true"
                                style={{ backgroundColor: entry.color }}
                            />
                            {entry.name}
                        </strong>
                        <dl>
                            <div>
                                <dt>{config.tooltipXLabel}</dt>
                                <dd>{formatRatio(point.x)}</dd>
                            </div>
                            <div>
                                <dt>{config.tooltipYLabel}</dt>
                                <dd>{formatRatio(point.y)}</dd>
                            </div>
                            <div>
                                <dt>Umbral</dt>
                                <dd>{formatThreshold(point.threshold)}</dd>
                            </div>
                        </dl>
                    </div>
                )
            })}
        </div>
    )
}

function CurveChart({ curveType, original, adversarial, scopeLabel, expanded = false }) {
    const config = CURVE_TYPES[curveType]
    const hasOriginal = isAvailable(original)
    const hasAdversarial = isAvailable(adversarial)
    const originalPoints = normalizeCurvePoints(original?.points)
    const adversarialPoints = normalizeCurvePoints(adversarial?.points)

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
        <div
            className={`curve-chart-wrap${expanded ? ' expanded' : ''}`}
            role="img"
            aria-label={`${config.label} original y adversarial para ${scopeLabel}`}
        >
            <ResponsiveContainer width="100%" height="100%">
                <LineChart margin={{ top: 12, right: 18, bottom: 8, left: 12 }}>
                    <CartesianGrid stroke="#e7edf5" strokeDasharray="4 4" />
                    <XAxis
                        dataKey="x"
                        type="number"
                        domain={[0, 1]}
                        ticks={TICKS}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        tickLine={false}
                        axisLine={{ stroke: '#94a3b8' }}
                        height={48}
                        label={{
                            value: config.xLabel,
                            position: 'insideBottom',
                            offset: -6,
                            fill: '#64748b',
                            fontSize: 13,
                            fontWeight: 750,
                        }}
                    />
                    <YAxis
                        type="number"
                        domain={[0, 1]}
                        ticks={TICKS}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        tick={{ fill: '#64748b', fontSize: 12 }}
                        tickLine={false}
                        axisLine={{ stroke: '#94a3b8' }}
                        width={72}
                        label={{
                            value: config.yLabel,
                            angle: -90,
                            position: 'insideLeft',
                            offset: 8,
                            fill: '#64748b',
                            fontSize: 13,
                            fontWeight: 750,
                        }}
                    />
                    <Tooltip
                        content={<CurveTooltip curveType={curveType} />}
                        cursor={{
                            stroke: '#94a3b8',
                            strokeDasharray: '4 4',
                        }}
                    />

                    {curveType === 'roc' && (
                        <ReferenceLine
                            segment={[
                                { x: 0, y: 0 },
                                { x: 1, y: 1 },
                            ]}
                            stroke="#94a3b8"
                            strokeWidth={2}
                            strokeDasharray="8 8"
                        />
                    )}

                    {hasOriginal && (
                        <Line
                            data={originalPoints}
                            dataKey="y"
                            name="Original"
                            type="linear"
                            stroke="var(--evaluation-original)"
                            strokeWidth={3}
                            dot={false}
                            activeDot={{ r: 4 }}
                            isAnimationActive={false}
                        />
                    )}

                    {hasAdversarial && (
                        <Line
                            data={adversarialPoints}
                            dataKey="y"
                            name="Adversarial"
                            type="linear"
                            stroke="var(--evaluation-adversarial)"
                            strokeWidth={3}
                            dot={false}
                            activeDot={{ r: 4 }}
                            isAnimationActive={false}
                        />
                    )}
                </LineChart>
            </ResponsiveContainer>
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
    const [isExpanded, setIsExpanded] = useState(false)
    const closeExpandedButtonRef = useRef(null)

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
    const scopeLabel = selectedScope?.class_reference?.label ?? 'Micro global'

    useEffect(() => {
        if (!isExpanded) {
            return undefined
        }

        const previousOverflow = document.body.style.overflow
        document.body.style.overflow = 'hidden'
        closeExpandedButtonRef.current?.focus()

        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                setIsExpanded(false)
            }
        }

        window.addEventListener('keydown', handleKeyDown)

        return () => {
            document.body.style.overflow = previousOverflow
            window.removeEventListener('keydown', handleKeyDown)
        }
    }, [isExpanded])

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
                        el ámbito micro global o una clase concreta.
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
                        <span>Ámbito de la curva</span>
                        <select
                            value={scope}
                            onChange={(event) => setScope(event.target.value)}
                        >
                            <option value="micro">Micro global (todas las clases)</option>
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
                <CurveScoreCard
                    label={`${config.scoreLabel} original`}
                    series={original}
                    tone="original"
                />
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
                        <h3>{scopeLabel}</h3>
                        <p>{config.description}</p>
                    </div>

                    <div className="curve-chart-actions">
                        <div className="evaluation-legend-pills" aria-label="Leyenda">
                            <span className="original">Original</span>
                            <span className="adversarial">Adversarial</span>
                            {curveType === 'roc' && (
                                <span className="reference">Azar</span>
                            )}
                        </div>

                        <button
                            type="button"
                            className="curve-expand-button"
                            onClick={() => setIsExpanded(true)}
                        >
                            <svg
                                aria-hidden="true"
                                viewBox="0 0 24 24"
                                width="16"
                                height="16"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <path d="M8 3H3v5" />
                                <path d="M16 3h5v5" />
                                <path d="M21 16v5h-5" />
                                <path d="M8 21H3v-5" />
                            </svg>

                            Ampliar gráfica
                        </button>
                    </div>
                </div>

                <CurveChart
                    curveType={curveType}
                    original={original}
                    adversarial={adversarial}
                    scopeLabel={scopeLabel}
                />
            </section>

            {isExpanded && (
                <div
                    className="curve-modal-backdrop"
                    role="presentation"
                    onMouseDown={(event) => {
                        if (event.target === event.currentTarget) {
                            setIsExpanded(false)
                        }
                    }}
                >
                    <section
                        className="curve-modal"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="curve-modal-title"
                    >
                        <header className="curve-modal-header">
                            <div>
                                <span className="evaluation-eyebrow">{config.label}</span>
                                <h2 id="curve-modal-title">{scopeLabel}</h2>
                            </div>

                            <div className="curve-modal-header-actions">
                                <div
                                    className="evaluation-legend-pills"
                                    aria-label="Leyenda"
                                >
                                    <span className="original">Original</span>
                                    <span className="adversarial">Adversarial</span>
                                    {curveType === 'roc' && (
                                        <span className="reference">Azar</span>
                                    )}
                                </div>

                                <button
                                    ref={closeExpandedButtonRef}
                                    type="button"
                                    className="curve-modal-close"
                                    onClick={() => setIsExpanded(false)}
                                >
                                    Cerrar
                                </button>
                            </div>
                        </header>

                        <div className="curve-modal-chart-frame">
                            <CurveChart
                                curveType={curveType}
                                original={original}
                                adversarial={adversarial}
                                scopeLabel={scopeLabel}
                                expanded
                            />
                        </div>
                    </section>
                </div>
            )}

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
