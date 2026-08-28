import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
    Bar,
    BarChart,
} from 'recharts'
import { resolveApiUrl } from '../../api/client'
import { generateEvaluationGradCam } from '../../api/evaluation'
import { generateDetectionEvaluationEigenCam } from '../../api/detectionEvaluation'
import {
    getClassificationClassComparison,
    getClassificationComparisonCaseSummary,
    getComparisonImagePairs,
    getDetectionClassComparison,
    getDetectionCocoComparisonRows,
    getDetectionObjectComparisonPairs,
    getDetectionTideComparisonRows,
    getDetectionObjectCaseSummary,
    getDetectionImageCaseSummary,
    matchesComparisonImageFilter,
    matchesDetectionObjectCaseFilter,
} from '../../utils/evaluationComparison'
import {
    clampEvaluationPreviewPan,
    clampEvaluationPreviewZoom,
    getEvaluationPreviewPanBounds,
    stepEvaluationPreviewZoom,
} from '../../utils/evaluationImagePreview'
import { TIDE_ERROR_DEFINITIONS } from '../../utils/detectionMetrics'
import { getEvaluationGradCamErrorMessage } from '../../utils/evaluationGradCamState'
import { getDetectionEigenCamErrorMessage } from '../../utils/detectionEvaluationEigenCamState'

const CLASSIFICATION_METRICS = [
    ['f1', 'F1'],
    ['recall', 'Recall'],
    ['precision', 'Precisión'],
    ['roc_auc', 'ROC-AUC'],
    ['average_precision', 'AP'],
]

function formatPercent(value, digits = 1) {
    return value == null || !Number.isFinite(Number(value))
        ? '—'
        : `${(Number(value) * 100).toFixed(digits)} %`
}

function formatPoints(value, digits = 1) {
    if (value == null || !Number.isFinite(Number(value))) return '—'
    const points = Number(value) * 100
    return `${points > 0 ? '+' : ''}${points.toFixed(digits)} pp`
}

function formatNumber(value, digits = 2) {
    return value == null || !Number.isFinite(Number(value))
        ? '—'
        : Number(value).toFixed(digits)
}

function ComparisonTabs({ tabs, activeTab, onChange }) {
    return (
        <nav className="evaluation-comparison-tabs" aria-label="Secciones de comparación">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    type="button"
                    className={activeTab === tab.id ? 'active' : ''}
                    aria-current={activeTab === tab.id ? 'page' : undefined}
                    onClick={() => onChange(tab.id)}
                >
                    {tab.label}
                    {tab.badge != null && <small>{tab.badge}</small>}
                </button>
            ))}
        </nav>
    )
}

function ComparisonSectionHeading({ eyebrow, title, description, controls }) {
    return (
        <header className="evaluation-comparison-section-heading">
            <div>
                {eyebrow && <span className="evaluation-eyebrow">{eyebrow}</span>}
                <h2>{title}</h2>
                {description && <p>{description}</p>}
            </div>
            {controls && <div className="evaluation-comparison-section-controls">{controls}</div>}
        </header>
    )
}

function EmptyComparison({ title, children }) {
    return (
        <div className="evaluation-comparison-empty-detail">
            <strong>{title}</strong>
            <span>{children}</span>
        </div>
    )
}

function ComparisonGuide({ intro, items = [], note }) {
    return (
        <details className="evaluation-comparison-guide">
            <summary>
                <span className="evaluation-comparison-guide-icon" aria-hidden="true">i</span>
                <span><strong>Guía de esta comparación</strong><small>Cómo interpretar esta vista A/B</small></span>
                <span className="evaluation-comparison-guide-chevron" aria-hidden="true">⌄</span>
            </summary>
            <div className="evaluation-comparison-guide-body">
                {intro && <p>{intro}</p>}
                {items.length > 0 && (
                    <div className="evaluation-comparison-guide-grid">
                        {items.map((item) => (
                            <div key={item.title}>
                                <strong>{item.title}</strong>
                                <span>{item.text}</span>
                            </div>
                        ))}
                    </div>
                )}
                {note && <p className="evaluation-comparison-guide-note"><strong>Clave de interpretación:</strong> {note}</p>}
            </div>
        </details>
    )
}

function ComparisonDataTable({ columns, rows, getKey = (row) => row.key }) {
    return (
        <div className="evaluation-comparison-data-table" role="table">
            <div
                className="evaluation-comparison-data-row header"
                role="row"
                style={{ '--comparison-columns': columns.map((column) => column.width ?? '1fr').join(' ') }}
            >
                {columns.map((column) => (
                    <span key={column.key} role="columnheader" className={column.className ?? ''}>
                        {column.label}
                    </span>
                ))}
            </div>
            {rows.map((row) => (
                <div
                    key={getKey(row)}
                    className="evaluation-comparison-data-row"
                    role="row"
                    style={{ '--comparison-columns': columns.map((column) => column.width ?? '1fr').join(' ') }}
                >
                    {columns.map((column) => (
                        <div key={column.key} role="cell" className={column.className ?? ''}>
                            {column.render ? column.render(row) : row[column.key]}
                        </div>
                    ))}
                </div>
            ))}
        </div>
    )
}

function interpolateCurveValue(points, x) {
    if (!points?.length || !Number.isFinite(x)) return null
    if (x <= points[0].x) return points[0].y
    if (x >= points[points.length - 1].x) return points[points.length - 1].y

    for (let index = 1; index < points.length; index += 1) {
        const right = points[index]
        const left = points[index - 1]
        if (x > right.x) continue
        if (Math.abs(right.x - left.x) < 1e-12) return right.y
        const weight = (x - left.x) / (right.x - left.x)
        return left.y + ((right.y - left.y) * weight)
    }
    return null
}

function ComparisonCurveTooltip({ active, payload, series, xLabel = 'X' }) {
    if (!active || !payload?.length) return null
    const guide = payload.find((entry) => entry.dataKey === 'guide')?.payload ?? payload[0]?.payload
    const x = Number(guide?.x)
    if (!Number.isFinite(x)) return null
    const values = series
        .map((entry) => ({ ...entry, value: interpolateCurveValue(entry.points, x) }))
        .filter((entry) => Number.isFinite(entry.value))
    if (!values.length) return null
    return (
        <div className="evaluation-chart-tooltip comparison-chart-tooltip comparison-curve-tooltip">
            <strong>{xLabel} ≈ {formatPercent(x, 1)}</strong>
            {values.map((entry) => (
                <div key={entry.key}>
                    <span style={{ '--series-color': entry.color }}>{entry.label}</span>
                    <strong>{formatPercent(entry.value, 1)}</strong>
                </div>
            ))}
        </div>
    )
}

function normalizeComparisonCurve(points, xKey = 'x') {
    return (points ?? [])
        .map((point) => ({
            x: Number(point[xKey]),
            y: Number(point.y ?? point.precision),
            threshold: point.threshold,
        }))
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
        .sort((left, right) => left.x - right.x)
}

function formatCurveScore(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—'
}

function ComparisonCurveChart({ series, yLabel = 'Valor', xLabel = 'Umbral', domain = [0, 1], scoreLabel = 'Área' }) {
    const normalizedSeries = series
        .map((entry) => ({ ...entry, points: normalizeComparisonCurve(entry.points, entry.xKey ?? 'x') }))
        .filter((entry) => entry.points.length > 0)

    if (!normalizedSeries.length) {
        return <EmptyComparison title="Curva no disponible">Las evaluaciones no contienen una curva representable para este ámbito.</EmptyComparison>
    }

    const guidePoints = Array.from({ length: 101 }, (_, index) => ({ x: index / 100, guide: 0 }))

    return (
        <>
            <div className="evaluation-comparison-curve-legend" aria-label="Leyenda de curvas">
                {normalizedSeries.map((entry) => (
                    <span key={entry.key} className={entry.reference ? 'reference' : ''}>
                        <i style={{ '--series-color': entry.color }} />
                        <strong>{entry.label}</strong>
                        {entry.score != null && <small>{scoreLabel} {formatCurveScore(entry.score)}</small>}
                    </span>
                ))}
            </div>
            <div className="evaluation-comparison-chart" role="img" aria-label="Comparación de curvas A y B">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart margin={{ top: 12, right: 28, bottom: 28, left: 18 }}>
                        <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="4 4" />
                        <XAxis
                            dataKey="x"
                            type="number"
                            domain={domain}
                            tickFormatter={(value) => Number(value).toFixed(2)}
                            tick={{ fill: 'var(--muted)', fontSize: 11 }}
                            label={{ value: xLabel, position: 'insideBottom', offset: -16, fill: 'var(--muted)' }}
                        />
                        <YAxis
                            type="number"
                            domain={domain}
                            tickFormatter={(value) => Number(value).toFixed(2)}
                            tick={{ fill: 'var(--muted)', fontSize: 11 }}
                            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 6, fill: 'var(--muted)' }}
                        />
                        <Tooltip
                            cursor={{ stroke: 'var(--comparison-reference)', strokeDasharray: '4 4' }}
                            wrapperStyle={{ zIndex: 50, pointerEvents: 'none' }}
                            content={<ComparisonCurveTooltip series={normalizedSeries} xLabel={xLabel} />}
                        />
                        <Line data={guidePoints} dataKey="guide" name="guide" stroke="transparent" dot={false} activeDot={false} isAnimationActive={false} />
                        {normalizedSeries.map((entry) => (
                            <Line
                                key={entry.key}
                                data={entry.points}
                                dataKey="y"
                                name={entry.label}
                                stroke={entry.color}
                                strokeWidth={entry.reference ? 2 : 3}
                                strokeDasharray={entry.reference ? '7 5' : undefined}
                                dot={false}
                                activeDot={false}
                                isAnimationActive={false}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </>
    )
}

function ClassDifferenceDivergingBars({ rows, leftKey, rightKey, label = 'Clase', limit = 12 }) {
    const data = [...(rows ?? [])]
        .filter((row) => Number.isFinite(row[leftKey]) && Number.isFinite(row[rightKey]))
        .map((row) => ({ ...row, comparisonDelta: row[rightKey] - row[leftKey] }))
        .sort((left, right) => Math.abs(right.comparisonDelta) - Math.abs(left.comparisonDelta))
        .slice(0, limit)
    if (!data.length) return null
    const maxDelta = Math.max(...data.map((row) => Math.abs(row.comparisonDelta)), 1e-9)
    return (
        <figure className="evaluation-comparison-diverging">
            <figcaption>
                <div><strong>Mayores diferencias</strong><span>Las categorías se ordenan por |B − A| para localizar primero dónde se separan más.</span></div>
                <div className="evaluation-comparison-diverging-axis"><span className="side-a">A tiene mayor valor</span><b>0</b><span className="side-b">B tiene mayor valor</span></div>
            </figcaption>
            <div className="evaluation-comparison-diverging-list">
                {data.map((row) => {
                    const delta = row.comparisonDelta
                    const width = `${Math.max(2, Math.abs(delta) / maxDelta * 100)}%`
                    return (
                        <div key={row.key} className="evaluation-comparison-diverging-row">
                            <strong title={row.label}>{row.label}</strong>
                            <div className="evaluation-comparison-diverging-track">
                                <span className="left">{delta < 0 && <i className="side-a" style={{ width }} />}</span>
                                <b aria-hidden="true" />
                                <span className="right">{delta > 0 && <i className="side-b" style={{ width }} />}</span>
                            </div>
                            <span className={delta < 0 ? 'side-a' : delta > 0 ? 'side-b' : ''}>{formatPoints(delta)}</span>
                        </div>
                    )
                })}
            </div>
            <small>Top {data.length} de {(rows ?? []).length} {label.toLowerCase()}s con valores comparables.</small>
        </figure>
    )
}

function splitCirclePath(cx, cy, r, side) {
    if (side === 'left') return `M ${cx} ${cy - r} A ${r} ${r} 0 0 0 ${cx} ${cy + r} Z`
    return `M ${cx} ${cy - r} A ${r} ${r} 0 0 1 ${cx} ${cy + r} Z`
}

function ClassDifferenceScatter({ rows, leftKey, rightKey, label = 'Clase' }) {
    const [scaleMode, setScaleMode] = useState('full')
    const rawPoints = (rows ?? []).filter((row) => Number.isFinite(row[leftKey]) && Number.isFinite(row[rightKey]))
    if (!rawPoints.length) return null
    const groups = Array.from(rawPoints.reduce((map, row) => {
        const left = Number(row[leftKey])
        const right = Number(row[rightKey])
        const key = `${left.toFixed(8)}:${right.toFixed(8)}`
        const current = map.get(key) ?? { key, left, right, rows: [], support: 0 }
        current.rows.push(row)
        current.support += Number(row.support ?? 0)
        map.set(key, current)
        return map
    }, new Map()).values())
    const dataMax = Math.max(...groups.flatMap((group) => [group.left, group.right]), 0)
    const domainMax = scaleMode === 'fit' ? Math.min(1, Math.max(0.1, dataMax * 1.12)) : 1
    const ticks = Array.from({ length: 5 }, (_, index) => domainMax * index / 4)
    const width = 680
    const height = 380
    const padding = { left: 66, right: 34, top: 28, bottom: 58 }
    const innerWidth = width - padding.left - padding.right
    const innerHeight = height - padding.top - padding.bottom
    const scaleX = (value) => padding.left + Math.max(0, Math.min(domainMax, value)) / domainMax * innerWidth
    const scaleY = (value) => height - padding.bottom - Math.max(0, Math.min(domainMax, value)) / domainMax * innerHeight
    return (
        <figure className="evaluation-comparison-scatter">
            <figcaption>
                <div><strong>Mapa A frente a B</strong><span>La diagonal representa igualdad. Las coordenadas idénticas se agrupan para no ocultar clases superpuestas.</span></div>
                <div className="evaluation-comparison-scatter-scale" role="group" aria-label="Escala del mapa de diferencias">
                    <button type="button" className={scaleMode === 'full' ? 'active' : ''} onClick={() => setScaleMode('full')}>Completa</button>
                    <button type="button" className={scaleMode === 'fit' ? 'active' : ''} onClick={() => setScaleMode('fit')}>Ajustar a datos</button>
                </div>
            </figcaption>
            <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Dispersión A frente a B por ${label.toLowerCase()}`}>
                {ticks.map((tick) => (
                    <g key={tick}>
                        <line x1={scaleX(tick)} x2={scaleX(tick)} y1={padding.top} y2={height - padding.bottom} className="grid" />
                        <line x1={padding.left} x2={width - padding.right} y1={scaleY(tick)} y2={scaleY(tick)} className="grid" />
                        <text x={scaleX(tick)} y={height - padding.bottom + 22} textAnchor="middle">{Math.round(tick * 100)}%</text>
                        <text x={padding.left - 12} y={scaleY(tick) + 4} textAnchor="end">{Math.round(tick * 100)}%</text>
                    </g>
                ))}
                <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={padding.top} className="diagonal" />
                {groups.map((group) => {
                    const cx = scaleX(group.left)
                    const cy = scaleY(group.right)
                    const r = Math.max(5, Math.min(9, 4 + Math.sqrt(Math.max(1, group.support)) * 0.28))
                    const equal = Math.abs(group.left - group.right) < 1e-10
                    const names = group.rows.slice(0, 6).map((row) => row.label).join(', ')
                    const more = group.rows.length > 6 ? ` y ${group.rows.length - 6} más` : ''
                    const title = `${group.rows.length > 1 ? `${group.rows.length} ${label.toLowerCase()}s` : group.rows[0].label}: A ${formatPercent(group.left)} · B ${formatPercent(group.right)}${group.rows.length > 1 ? ` · ${names}${more}` : ''}`
                    return (
                        <g key={group.key} className="evaluation-comparison-scatter-point">
                            <title>{title}</title>
                            {equal ? <>
                                <path d={splitCirclePath(cx, cy, r, 'left')} className="side-a equal-half" />
                                <path d={splitCirclePath(cx, cy, r, 'right')} className="side-b equal-half" />
                            </> : <circle cx={cx} cy={cy} r={r} className={group.right > group.left ? 'side-b' : 'side-a'} />}
                            {group.rows.length > 1 && <text x={cx + r + 4} y={cy - r - 2} className="cluster-count">×{group.rows.length}</text>}
                        </g>
                    )
                })}
                <text x={padding.left + innerWidth / 2} y={height - 8} textAnchor="middle" className="axis-label">Evaluación A</text>
                <text x="15" y={padding.top + innerHeight / 2} textAnchor="middle" transform={`rotate(-90 15 ${padding.top + innerHeight / 2})`} className="axis-label">Evaluación B</text>
            </svg>
            <div className="evaluation-comparison-scatter-legend"><span className="side-a">Por debajo: A tiene mayor valor</span><span>◐ igualdad A/B · ×n clases coincidentes</span><span className="side-b">Por encima: B tiene mayor valor</span></div>
        </figure>
    )
}

function ClassificationClassComparison({ leftEvaluation, rightEvaluation }) {
    const [metric, setMetric] = useState('f1')
    const [query, setQuery] = useState('')
    const [sort, setSort] = useState('difference')
    const rows = useMemo(() => {
        const normalized = query.trim().toLocaleLowerCase('es')
        const data = getClassificationClassComparison(leftEvaluation, rightEvaluation, metric)
            .filter((row) => !normalized || row.label.toLocaleLowerCase('es').includes(normalized))
        return [...data].sort((left, right) => {
            if (sort === 'name') return left.label.localeCompare(right.label, 'es')
            if (sort === 'support') return right.support - left.support
            return Math.abs(right.delta ?? -Infinity) - Math.abs(left.delta ?? -Infinity)
        })
    }, [leftEvaluation, metric, query, rightEvaluation, sort])

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading
                eyebrow="Diagnóstico por clase"
                title="Dónde difieren A y B"
                description="Compara el rendimiento robusto de la misma clase. Las barras permiten detectar el patrón y el porcentaje conserva el valor exacto."
                controls={(
                    <>
                        <label><span>Métrica</span><select value={metric} onChange={(event) => setMetric(event.target.value)}>{CLASSIFICATION_METRICS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
                        <label><span>Orden</span><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="difference">Mayor diferencia</option><option value="support">Mayor soporte</option><option value="name">Nombre</option></select></label>
                    </>
                )}
            />
            <ComparisonGuide
                intro="Lee primero la distancia entre las barras A y B y usa después el valor exacto para cuantificarla."
                items={[
                    { title: 'Mayor diferencia', text: 'Prioriza las clases donde las dos evaluaciones divergen más en la métrica seleccionada.' },
                    { title: 'Soporte', text: 'Indica cuántas muestras reales tiene la clase; diferencias grandes con poco soporte deben interpretarse con cautela.' },
                ]}
                note="El delta es B − A. Un valor positivo no significa automáticamente una mejora si eliges una métrica donde menor sea mejor."
            />
            <div className="evaluation-comparison-search-row">
                <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar clase" />
                <span>{rows.length} clases</span>
            </div>
            <ClassDifferenceDivergingBars rows={rows} leftKey="leftRobust" rightKey="rightRobust" label="Clase" />
            <ClassDifferenceScatter rows={rows} leftKey="leftRobust" rightKey="rightRobust" label="Clase" />
            <div className="evaluation-comparison-class-list">
                {rows.map((row) => (
                    <article key={row.key} className="evaluation-comparison-class-row">
                        <header><div><strong>{row.label}</strong><small>Soporte {row.support}</small></div><span>{formatPoints(row.delta)}</span></header>
                        <div className="evaluation-comparison-class-bars">
                            <div><span className="side-a">A</span><i><b className="side-a" style={{ width: `${Math.max(0, Math.min(1, row.leftRobust ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row.leftRobust)}</strong></div>
                            <div><span className="side-b">B</span><i><b className="side-b" style={{ width: `${Math.max(0, Math.min(1, row.rightRobust ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row.rightRobust)}</strong></div>
                        </div>
                    </article>
                ))}
            </div>
        </section>
    )
}

function getClassificationCurve(evaluation, curveType) {
    return evaluation?.advanced_metrics?.supervised?.[curveType]?.micro ?? null
}

function ClassificationCurveComparison({ leftEvaluation, rightEvaluation, sameModel }) {
    const [curveType, setCurveType] = useState('roc')
    const left = getClassificationCurve(leftEvaluation, curveType)
    const right = getClassificationCurve(rightEvaluation, curveType)
    const series = [
        ...(sameModel && left?.original?.available ? [{ key: 'reference', label: 'Referencia original común', color: 'var(--comparison-reference)', reference: true, points: left.original.points, score: left.original.score }] : []),
        ...(!sameModel && left?.original?.available ? [{ key: 'a-original', label: 'A · original', color: 'var(--comparison-a)', reference: true, points: left.original.points, score: left.original.score }] : []),
        ...(!sameModel && right?.original?.available ? [{ key: 'b-original', label: 'B · original', color: 'var(--comparison-b)', reference: true, points: right.original.points, score: right.original.score }] : []),
        ...(left?.adversarial?.available ? [{ key: 'a', label: 'A · robusta', color: 'var(--comparison-a)', points: left.adversarial.points, score: left.adversarial.score }] : []),
        ...(right?.adversarial?.available ? [{ key: 'b', label: 'B · robusta', color: 'var(--comparison-b)', points: right.adversarial.points, score: right.adversarial.score }] : []),
    ]

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading
                eyebrow="Análisis por umbral"
                title={curveType === 'roc' ? 'Curvas ROC A frente a B' : 'Curvas Precision–Recall A frente a B'}
                description="Las series se dibujan con sus propios puntos: no necesitan compartir exactamente la misma rejilla de umbrales."
                controls={<div className="evaluation-comparison-segmented"><button className={curveType === 'roc' ? 'active' : ''} onClick={() => setCurveType('roc')} type="button">ROC</button><button className={curveType === 'precision_recall' ? 'active' : ''} onClick={() => setCurveType('precision_recall')} type="button">Precision–Recall</button></div>}
            />
            <ComparisonGuide
                intro="El color identifica la evaluación. La línea discontinua representa la referencia original y la continua el resultado tras el ataque."
                items={[
                    { title: curveType === 'roc' ? 'ROC' : 'Precision–Recall', text: curveType === 'roc' ? 'Una curva más próxima a la esquina superior izquierda discrimina mejor entre clases.' : 'Una curva más alta conserva más precisión para el mismo nivel de recall.' },
                    { title: sameModel ? 'Referencia común' : 'Dos referencias', text: sameModel ? 'Como el modelo y dataset son los mismos, se dibuja una sola referencia original.' : 'Al comparar modelos distintos cada evaluación conserva su propia referencia original.' },
                    { title: 'Inspección comparable', text: 'El tooltip usa la misma posición del eje X para estimar A y B. Las curvas conservan sus puntos reales y no se suavizan.' },
                ]}
                note={curveType === 'precision_recall' ? 'Las irregularidades de la curva reflejan los puntos empíricos almacenados; no se alisan para mejorar artificialmente su aspecto.' : undefined}
            />
            <ComparisonCurveChart series={series} xLabel={curveType === 'roc' ? 'Falsos positivos' : 'Recall'} yLabel={curveType === 'roc' ? 'Verdaderos positivos' : 'Precisión'} scoreLabel={curveType === 'roc' ? 'AUC' : 'AP'} />
        </section>
    )
}

function DetectionCocoComparison({ leftEvaluation, rightEvaluation }) {
    const rows = getDetectionCocoComparisonRows(leftEvaluation, rightEvaluation)
    const highlights = rows.filter((row) => ['map_50_95', 'ap_50', 'ap_75', 'ar_100'].includes(row.key))
    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Protocolo COCO" title="Métricas robustas A frente a B" description="Primero compara la longitud de las barras; la tabla conserva después los valores exactos del protocolo COCO." />
            <ComparisonGuide
                intro="AP y AR miden calidad de detección, no intensidad del ataque. Una barra más larga significa mayor rendimiento robusto bajo esa evaluación concreta."
                items={[
                    { title: 'AP', text: 'Resume precisión a lo largo de la curva Precision–Recall. AP@0.50 es más permisivo y AP@0.75 exige una localización más precisa.' },
                    { title: 'AR', text: 'Mide qué fracción de los objetos de referencia puede recuperarse permitiendo un máximo de detecciones por imagen.' },
                ]}
                note="A y B identifican evaluaciones. El delta B − A cuantifica la diferencia, pero no implica que los ataques tengan una intensidad equivalente."
            />
            <div className="evaluation-comparison-coco-visual" aria-label="Resumen visual de métricas COCO">
                {highlights.map((row) => (
                    <article key={row.key}>
                        <header><strong>{row.label}</strong><span>{formatPoints(row.delta)}</span></header>
                        <div className="evaluation-comparison-class-bars">
                            <div><span className="side-a">A</span><i><b className="side-a" style={{ width: `${Math.max(0, Math.min(1, row.left ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row.left)}</strong></div>
                            <div><span className="side-b">B</span><i><b className="side-b" style={{ width: `${Math.max(0, Math.min(1, row.right ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row.right)}</strong></div>
                        </div>
                    </article>
                ))}
            </div>
            <details className="evaluation-comparison-technical-details">
                <summary>Ver valores COCO detallados <span>{rows.length} métricas</span></summary>
                <ComparisonDataTable
                    columns={[
                        { key: 'label', label: 'Métrica', width: '1.6fr', render: (row) => <strong>{row.label}</strong> },
                        { key: 'left', label: 'A', className: 'side-a', render: (row) => formatPercent(row.left) },
                        { key: 'right', label: 'B', className: 'side-b', render: (row) => formatPercent(row.right) },
                        { key: 'delta', label: 'Δ B − A', render: (row) => formatPoints(row.delta) },
                    ]}
                    rows={rows}
                />
            </details>
        </section>
    )
}

function DetectionClassComparison({ leftEvaluation, rightEvaluation }) {
    const [query, setQuery] = useState('')
    const [sort, setSort] = useState('difference')
    const [metric, setMetric] = useState('ap')
    const metricConfig = {
        ap: { label: 'AP', leftKey: 'leftAp', rightKey: 'rightAp' },
        ar: { label: 'AR', leftKey: 'leftAr', rightKey: 'rightAr' },
        preservation: { label: 'Preservación', leftKey: 'leftPreservationRate', rightKey: 'rightPreservationRate' },
        suppression: { label: 'Supresión', leftKey: 'leftSuppressionRate', rightKey: 'rightSuppressionRate' },
    }[metric]
    const rows = useMemo(() => {
        const normalized = query.trim().toLocaleLowerCase('es')
        return getDetectionClassComparison(leftEvaluation, rightEvaluation)
            .map((row) => ({
                ...row,
                selectedDelta: Number.isFinite(row[metricConfig.leftKey]) && Number.isFinite(row[metricConfig.rightKey])
                    ? row[metricConfig.rightKey] - row[metricConfig.leftKey]
                    : null,
            }))
            .filter((row) => !normalized || row.label.toLocaleLowerCase('es').includes(normalized))
            .sort((left, right) => sort === 'support'
                ? right.support - left.support
                : sort === 'name'
                    ? left.label.localeCompare(right.label, 'es')
                    : Math.abs(right.selectedDelta ?? -Infinity) - Math.abs(left.selectedDelta ?? -Infinity))
    }, [leftEvaluation, metricConfig.leftKey, metricConfig.rightKey, query, rightEvaluation, sort])

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading
                eyebrow="Desglose por categoría"
                title={`${metricConfig.label} por clase: dónde difieren A y B`}
                description="El gráfico divergente prioriza las categorías con mayor separación; el mapa A/B conserva la visión global y puede ajustar su escala a los datos."
                controls={<>
                    <label><span>Métrica</span><select value={metric} onChange={(event) => setMetric(event.target.value)}><option value="ap">AP</option><option value="ar">AR</option><option value="preservation">Preservación</option><option value="suppression">Supresión</option></select></label>
                    <label><span>Orden</span><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="difference">Mayor diferencia</option><option value="support">Mayor soporte</option><option value="name">Nombre</option></select></label>
                </>}
            />
            <ComparisonGuide
                intro="Empieza por las mayores diferencias y utiliza el mapa A/B para comprobar si el patrón es general o se concentra en pocas categorías."
                items={[
                    { title: 'AP / AR', text: 'AP resume calidad Precision–Recall y AR la recuperación de objetos a través del protocolo COCO.' },
                    { title: 'Preservación / supresión', text: 'Miden qué proporción de los objetos detectados originalmente se mantiene o desaparece tras el ataque.' },
                ]}
                note="Los colores identifican A y B. En supresión, un valor mayor significa más objetos perdidos; la vista no lo presenta como un ganador automático."
            />
            <div className="evaluation-comparison-search-row"><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar categoría" /><span>{rows.length} clases</span></div>
            <ClassDifferenceDivergingBars rows={rows} leftKey={metricConfig.leftKey} rightKey={metricConfig.rightKey} label="Categoría" />
            <ClassDifferenceScatter rows={rows} leftKey={metricConfig.leftKey} rightKey={metricConfig.rightKey} label="Categoría" />
            <div className="evaluation-comparison-class-list">
                {rows.map((row) => (
                    <article key={row.key} className="evaluation-comparison-class-row detection">
                        <header><div><strong>{row.label}</strong><small>{row.support} objetos</small></div><span>Δ {metricConfig.label} {formatPoints(row.selectedDelta)}</span></header>
                        <div className="evaluation-comparison-class-bars">
                            <div><span className="side-a">A</span><i><b className="side-a" style={{ width: `${Math.max(0, Math.min(1, row[metricConfig.leftKey] ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row[metricConfig.leftKey])}</strong></div>
                            <div><span className="side-b">B</span><i><b className="side-b" style={{ width: `${Math.max(0, Math.min(1, row[metricConfig.rightKey] ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row[metricConfig.rightKey])}</strong></div>
                        </div>
                        <dl className="evaluation-comparison-class-meta"><div><dt>AP A / B</dt><dd>{formatPercent(row.leftAp)} / {formatPercent(row.rightAp)}</dd></div><div><dt>AR A / B</dt><dd>{formatPercent(row.leftAr)} / {formatPercent(row.rightAr)}</dd></div><div><dt>Preservación A / B</dt><dd>{formatPercent(row.leftPreservationRate)} / {formatPercent(row.rightPreservationRate)}</dd></div><div><dt>Suprimidos A / B</dt><dd>{row.leftSuppressed} / {row.rightSuppressed}</dd></div></dl>
                    </article>
                ))}
            </div>
        </section>
    )
}

function getDetectionPrSeries(evaluation, scope) {
    const supervised = evaluation?.supervised
    if (!supervised?.available) return null
    if (scope === 'global') {
        return {
            curve: supervised.adversarial?.precision_recall_curve,
            score: supervised.adversarial?.aggregate?.map_50_95,
        }
    }
    const row = supervised.adversarial?.per_class?.find((entry) => String(entry.reference?.category_internal_id ?? entry.reference?.category_name) === scope)
    return row ? { curve: row.precision_recall_curve, score: row.average_precision } : null
}

function DetectionCurveComparison({ leftEvaluation, rightEvaluation }) {
    const classes = leftEvaluation?.supervised?.adversarial?.per_class ?? []
    const [scope, setScope] = useState('global')
    const leftCurve = getDetectionPrSeries(leftEvaluation, scope)
    const rightCurve = getDetectionPrSeries(rightEvaluation, scope)
    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Precision–Recall" title="Curvas robustas A frente a B" description="Compara la envolvente Precision–Recall del promedio COCO global o de una categoría concreta." controls={<label><span>Ámbito</span><select value={scope} onChange={(event) => setScope(event.target.value)}><option value="global">Promedio COCO global</option>{classes.map((entry) => { const key = String(entry.reference?.category_internal_id ?? entry.reference?.category_name); return <option key={key} value={key}>{entry.reference?.category_name}</option> })}</select></label>} />
            <ComparisonGuide intro="Una curva más alta mantiene más precisión para el mismo recall. A y B usan colores de identidad, no de calidad." items={[{ title: 'Promedio global', text: 'Promedia el protocolo COCO sobre clases e IoU; una clase concreta permite localizar diferencias específicas.' }, { title: 'Área de la curva', text: 'Una curva superior suele corresponder a mayor AP.' }]} />
            <ComparisonCurveChart
                series={[
                    { key: 'a', label: 'Evaluación A', color: 'var(--comparison-a)', points: leftCurve?.curve?.points, xKey: 'recall', score: leftCurve?.score },
                    { key: 'b', label: 'Evaluación B', color: 'var(--comparison-b)', points: rightCurve?.curve?.points, xKey: 'recall', score: rightCurve?.score },
                ]}
                xLabel="Recall"
                yLabel="Precisión"
                scoreLabel={scope === 'global' ? 'mAP' : 'AP'}
            />
        </section>
    )
}

function ComparisonTideTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null
    return (
        <div className="evaluation-chart-tooltip comparison-chart-tooltip tide-tooltip">
            <strong>{label}</strong>
            {payload.map((entry) => (
                <div key={`${entry.name}-${entry.dataKey}`}>
                    <span style={{ '--series-color': entry.color }}>{entry.name}</span>
                    <strong>{formatNumber(entry.value, 1)} pp</strong>
                </div>
            ))}
        </div>
    )
}

function DetectionTideComparison({ leftEvaluation, rightEvaluation }) {
    const rows = getDetectionTideComparisonRows(leftEvaluation, rightEvaluation).map((row) => ({
        ...row,
        label: TIDE_ERROR_DEFINITIONS[row.key]?.label ?? row.key,
        shortLabel: TIDE_ERROR_DEFINITIONS[row.key]?.shortLabel ?? row.key,
    }))
    if (!rows.length) return <EmptyComparison title="TIDE no disponible">Una de las evaluaciones no contiene diagnóstico TIDE.</EmptyComparison>
    const chartData = rows.map((row) => ({ label: row.shortLabel, a: row.leftImpact, b: row.rightImpact }))
    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Diagnóstico TIDE" title="Qué tipo de error pesa más en A y B" description="La gráfica ofrece la lectura inmediata y la tabla conserva el valor exacto y la frecuencia de cada tipo de error." />
            <ComparisonGuide intro="dAP mide cuánto AP50 se recuperaría corrigiendo hipotéticamente un tipo de error. Una barra más larga indica mayor impacto, no necesariamente más casos." items={[{ title: 'Impacto (dAP)', text: 'Cuantifica cuánto perjudica el error al AP50.' }, { title: 'Frecuencia', text: 'Cuenta cuántas veces aparece el error; frecuencia e impacto no son equivalentes.' }]} />
            <div className="evaluation-comparison-chart tide" role="img" aria-label="Comparación TIDE A y B">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 24, bottom: 18, left: 12 }}>
                        <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="4 4" horizontal={false} />
                        <XAxis type="number" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                        <YAxis type="category" dataKey="label" width={120} tick={{ fill: 'var(--text-soft)', fontSize: 11 }} />
                        <Tooltip cursor={false} wrapperStyle={{ zIndex: 50, pointerEvents: 'none' }} content={<ComparisonTideTooltip />} />
                        <Bar dataKey="a" name="Evaluación A" fill="var(--comparison-a)" radius={[0, 5, 5, 0]} />
                        <Bar dataKey="b" name="Evaluación B" fill="var(--comparison-b)" radius={[0, 5, 5, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            <ComparisonDataTable columns={[
                { key: 'label', label: 'Error', width: '1.4fr', render: (row) => <strong>{row.label}</strong> },
                { key: 'leftImpact', label: 'dAP A', className: 'side-a', render: (row) => `${formatNumber(row.leftImpact, 1)} pp` },
                { key: 'rightImpact', label: 'dAP B', className: 'side-b', render: (row) => `${formatNumber(row.rightImpact, 1)} pp` },
                { key: 'counts', label: 'Casos A / B', render: (row) => `${row.leftCount} / ${row.rightCount ?? '—'}` },
            ]} rows={rows} />
        </section>
    )
}

const DETECTION_OUTCOME_LABELS = {
    preserved: 'Preservado',
    suppressed: 'Suprimido',
    recovered: 'Recuperado',
    missed_both: 'No detectado en ambas',
}

function detectionObjectOverlay(object, phase, includeReference = true) {
    if (!object) return []
    const detections = []
    if (includeReference && object.ground_truth_box) {
        detections.push({ bbox: object.ground_truth_box, label: `Referencia · ${object.category?.category_name ?? 'objeto'}`, tone: 'reference' })
    }
    const observation = phase === 'original' ? object.original_observation : object.adversarial_observation
    if (observation?.detection_box) {
        detections.push({
            bbox: observation.detection_box,
            label: object.category?.category_name ?? 'objeto',
            confidence: observation.confidence,
            tone: phase === 'original' ? 'original' : 'adversarial',
        })
    }
    return detections
}

function DetectionObjectCases({ leftEvaluation, rightEvaluation }) {
    const pairs = useMemo(() => getDetectionObjectComparisonPairs(leftEvaluation, rightEvaluation), [leftEvaluation, rightEvaluation])
    const summary = useMemo(() => getDetectionObjectCaseSummary(leftEvaluation, rightEvaluation), [leftEvaluation, rightEvaluation])
    const [filter, setFilter] = useState('different')
    const [visible, setVisible] = useState(8)
    const [preview, setPreview] = useState(null)
    const leftResults = useMemo(() => new Map((leftEvaluation?.results ?? []).map((row) => [String(row.image_id), row])), [leftEvaluation])
    const rightResults = useMemo(() => new Map((rightEvaluation?.results ?? []).map((row) => [String(row.image_id), row])), [rightEvaluation])
    const filtered = useMemo(() => pairs.filter((pair) => matchesDetectionObjectCaseFilter(pair, filter)), [filter, pairs])
    const shown = filtered.slice(0, visible)
    const filters = [
        ['different', 'A y B discrepan', summary.different],
        ['only_a', 'Solo A preserva', summary.only_a],
        ['only_b', 'Solo B preserva', summary.only_b],
        ['both_preserved', 'Ambas preservan', summary.both_preserved],
        ['both_suppressed', 'Ambas suprimen', summary.both_suppressed],
        ['all', 'Todos', summary.total],
    ]

    if (!pairs.length) return <EmptyComparison title="Detalle por objeto no disponible">Las evaluaciones no contienen objetos materializados comunes.</EmptyComparison>

    return (
        <section className="evaluation-comparison-detail-section evaluation-comparison-object-cases">
            <ComparisonSectionHeading eyebrow="Casos clave" title="Objetos que explican la diferencia" description="En lugar de leer coordenadas, inspecciona la referencia y el resultado visual del mismo objeto bajo A y B." />
            <ComparisonGuide
                intro="Empieza por los objetos donde A y B discrepan. Cada caso conserva la caja de referencia y superpone la detección asociada cuando existe."
                items={[
                    { title: 'Solo A / solo B preserva', text: 'Aísla objetos que sobreviven al ataque únicamente en una de las evaluaciones.' },
                    { title: 'Referencia', text: 'La caja de referencia procede de la anotación del dataset y sirve de ancla para interpretar IoU y desapariciones.' },
                ]}
                note="Los IDs y coordenadas siguen disponibles en los datos almacenados, pero la vista prioriza la evidencia visual porque es más interpretable."
            />
            <div className="evaluation-comparison-paired-stats" aria-label="Resumen de preservación A/B">
                <div><span>Desacuerdo de preservación</span><strong>{summary.preservation_disagreements} / {summary.total}</strong><small>{formatPercent(summary.preservation_disagreement_rate)} de los objetos comparables</small></div>
                <div><span>Balance neto</span><strong className={summary.preservation_balance < 0 ? 'side-a' : summary.preservation_balance > 0 ? 'side-b' : ''}>{summary.preservation_balance > 0 ? '+' : ''}{summary.preservation_balance}</strong><small>{summary.preservation_balance === 0 ? 'Mismo número preservado en exclusiva' : `${Math.abs(summary.preservation_balance)} objetos hacia ${summary.preservation_balance > 0 ? 'B' : 'A'}`}</small></div>
            </div>
            <div className="evaluation-comparison-case-summary" role="group" aria-label="Filtros de comportamiento por objeto">
                {filters.map(([id, label, count]) => (
                    <button key={id} type="button" className={filter === id ? 'active' : ''} onClick={() => { setFilter(id); setVisible(8) }}>
                        <strong>{count}</strong><span>{label}</span>
                    </button>
                ))}
            </div>
            <div className="evaluation-comparison-object-evidence-list">
                {shown.map(({ left, right }) => {
                    const leftResult = leftResults.get(String(left.image_id))
                    const rightResult = rightResults.get(String(right.image_id))
                    const originalUrl = `/datasets/${leftEvaluation.dataset_id}/images/${left.image_id}`
                    return (
                        <article key={left.object_id} className="evaluation-comparison-object-evidence">
                            <header>
                                <div><span className="evaluation-eyebrow">{left.category?.category_name ?? 'Objeto'}</span><strong>{leftResult?.filename ?? left.image_id}</strong><small>{left.object_id}</small></div>
                                <div><span className="side-a">A · {DETECTION_OUTCOME_LABELS[left.outcome] ?? left.outcome}</span><span className="side-b">B · {DETECTION_OUTCOME_LABELS[right.outcome] ?? right.outcome}</span></div>
                            </header>
                            <div className="evaluation-comparison-object-visual-grid">
                                <ComparisonImageFigure label="Referencia" url={originalUrl} caption={`${left.category?.category_name ?? 'objeto'} · ${left.size_bucket ?? 'tamaño n/d'}`} detections={detectionObjectOverlay(left, 'original', true).filter((item) => item.tone === 'reference')} onExpand={setPreview} tone="reference" />
                                <ComparisonImageFigure label="A · Tras el ataque" url={leftResult?.adversarial_image_url ?? originalUrl} caption={`${DETECTION_OUTCOME_LABELS[left.outcome] ?? left.outcome} · conf. ${formatNumber(left.adversarial_observation?.confidence)} · IoU ${formatNumber(left.adversarial_observation?.iou_with_ground_truth)}`} detections={detectionObjectOverlay(left, 'adversarial')} onExpand={setPreview} tone="side-a" />
                                <ComparisonImageFigure label="B · Tras el ataque" url={rightResult?.adversarial_image_url ?? originalUrl} caption={`${DETECTION_OUTCOME_LABELS[right.outcome] ?? right.outcome} · conf. ${formatNumber(right.adversarial_observation?.confidence)} · IoU ${formatNumber(right.adversarial_observation?.iou_with_ground_truth)}`} detections={detectionObjectOverlay(right, 'adversarial')} onExpand={setPreview} tone="side-b" />
                            </div>
                        </article>
                    )
                })}
            </div>
            {!shown.length && <EmptyComparison title="No hay objetos en este grupo">Selecciona otro patrón de comportamiento.</EmptyComparison>}
            {filtered.length > 8 && <div className="evaluation-comparison-more"><span>{Math.min(visible, filtered.length)} / {filtered.length}</span>{visible < filtered.length ? <button type="button" onClick={() => setVisible((current) => Math.min(filtered.length, current + 8))}>Ver 8 más</button> : <button type="button" onClick={() => setVisible(8)}>Mostrar menos</button>}</div>}
            {preview && <ComparisonImageLightbox preview={preview} onClose={() => setPreview(null)} />}
        </section>
    )
}

function ExpandIcon() {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 4H4v4.5M15.5 4H20v4.5M20 15.5V20h-4.5M8.5 20H4v-4.5" /></svg>
}

function ZoomOutIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14" /></svg> }
function ZoomInIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg> }
function CloseIcon() { return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg> }

function DetectionLabelsOverlay({ detections = [], dimensions }) {
    if (!dimensions || !detections?.length) return null
    const fontSize = Math.max(12, Math.round(Math.min(dimensions.width, dimensions.height) * 0.024))
    const strokeWidth = Math.max(2, Math.round(Math.min(dimensions.width, dimensions.height) * 0.004))
    return (
        <svg className="evaluation-comparison-detection-overlay" viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} aria-label={`${detections.length} detecciones`} role="img">
            {detections.map((detection, index) => {
                const box = detection.bbox
                if (!box) return null
                const confidence = Number.isFinite(Number(detection.confidence)) ? ` · ${formatNumber(detection.confidence, 2)}` : ''
                const label = `${detection.label ?? 'objeto'}${confidence}`
                const labelWidth = Math.min(box.x2 - box.x1, Math.max(fontSize * 4.5, label.length * fontSize * 0.58))
                const labelY = Math.max(0, box.y1 - fontSize * 1.45)
                return (
                    <g key={`${detection.label ?? 'detection'}-${index}`}>
                        <rect className={`evaluation-comparison-detection-box ${detection.tone ?? ''}`} x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1} strokeWidth={strokeWidth} />
                        <rect className={`evaluation-comparison-detection-label-bg ${detection.tone ?? ''}`} x={box.x1} y={labelY} width={labelWidth} height={fontSize * 1.45} />
                        <text className={`evaluation-comparison-detection-label ${detection.tone ?? ''}`} x={box.x1 + fontSize * 0.25} y={labelY + fontSize} fontSize={fontSize}>{label}</text>
                    </g>
                )
            })}
        </svg>
    )
}

function ComparisonImageLightbox({ preview, onClose }) {
    const [zoom, setZoom] = useState(1)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [dragging, setDragging] = useState(false)
    const [dimensions, setDimensions] = useState(null)
    const viewportRef = useRef(null)
    const canvasRef = useRef(null)
    const dragRef = useRef(null)

    useEffect(() => {
        const previous = document.body.style.overflow
        document.body.style.overflow = 'hidden'
        const handler = (event) => event.key === 'Escape' && onClose()
        window.addEventListener('keydown', handler)
        return () => { document.body.style.overflow = previous; window.removeEventListener('keydown', handler) }
    }, [onClose])

    function bounds(nextZoom = zoom) {
        const viewport = viewportRef.current
        const canvas = canvasRef.current
        if (!viewport || !canvas) return { x: 0, y: 0 }
        return getEvaluationPreviewPanBounds({ viewportWidth: viewport.clientWidth, viewportHeight: viewport.clientHeight, imageWidth: canvas.offsetWidth, imageHeight: canvas.offsetHeight, zoom: nextZoom })
    }
    function changeZoom(direction) {
        const next = stepEvaluationPreviewZoom(zoom, direction)
        setZoom(next)
        setPan((current) => clampEvaluationPreviewPan(current, bounds(next)))
    }
    function reset() { setZoom(1); setPan({ x: 0, y: 0 }); setDragging(false); dragRef.current = null }
    function pointerDown(event) {
        if (zoom <= 1) return
        event.currentTarget.setPointerCapture(event.pointerId)
        dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y }
        setDragging(true)
    }
    function pointerMove(event) {
        const drag = dragRef.current
        if (!drag || drag.id !== event.pointerId || zoom <= 1) return
        setPan(clampEvaluationPreviewPan({ x: drag.panX + event.clientX - drag.x, y: drag.panY + event.clientY - drag.y }, bounds()))
    }
    function pointerUp(event) {
        const drag = dragRef.current
        if (!drag || drag.id !== event.pointerId) return
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
        dragRef.current = null
        setDragging(false)
    }

    const safeZoom = clampEvaluationPreviewZoom(zoom)
    const modal = (
        <div className="image-lightbox-backdrop evaluation-image-lightbox-backdrop comparison-image-lightbox-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
            <section className="image-lightbox evaluation-image-lightbox comparison-image-lightbox" role="dialog" aria-modal="true" aria-label={`Vista ampliada: ${preview.label}`}>
                <header>
                    <div><span className="evaluation-eyebrow">Vista ampliada</span><strong>{preview.label}</strong></div>
                    <div className="evaluation-image-lightbox-actions">
                        <button type="button" className="image-lightbox-close evaluation-image-zoom-button" onClick={() => changeZoom(-1)} disabled={safeZoom <= 1} aria-label="Alejar imagen"><ZoomOutIcon /></button>
                        <output>{Math.round(safeZoom * 100)}%</output>
                        <button type="button" className="image-lightbox-close evaluation-image-zoom-button" onClick={() => changeZoom(1)} disabled={safeZoom >= 4} aria-label="Acercar imagen"><ZoomInIcon /></button>
                        <button type="button" className="secondary-action evaluation-image-reset-button" onClick={reset} disabled={safeZoom === 1 && pan.x === 0 && pan.y === 0}>Restablecer</button>
                        <button type="button" className="image-lightbox-close" onClick={onClose} aria-label="Cerrar vista ampliada"><CloseIcon /></button>
                    </div>
                </header>
                <div ref={viewportRef} className={`evaluation-image-lightbox-stage${safeZoom > 1 ? ' can-pan' : ''}${dragging ? ' is-dragging' : ''}`} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp}>
                    <div ref={canvasRef} className="evaluation-comparison-lightbox-canvas" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${safeZoom})` }}>
                        <img src={resolveApiUrl(preview.url)} alt={preview.label} draggable="false" onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
                        <DetectionLabelsOverlay detections={preview.detections} dimensions={dimensions} />
                    </div>
                </div>
                <footer><span>Usa + / − para cambiar el zoom. Con zoom activo, arrastra para inspeccionar otras zonas.</span></footer>
            </section>
        </div>
    )
    return createPortal(modal, document.body)
}

function ComparisonImageFigure({ label, url, caption, onExpand, tone, detections = [] }) {
    const [dimensions, setDimensions] = useState(null)
    return (
        <figure className={`evaluation-comparison-image-figure ${tone ?? ''}`}>
            <figcaption><strong>{label}</strong>{caption && <span>{caption}</span>}</figcaption>
            <div>
                <img src={resolveApiUrl(url)} alt={label} loading="lazy" onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
                <DetectionLabelsOverlay detections={detections} dimensions={dimensions} />
                <button type="button" onClick={() => onExpand({ label, url, detections })} aria-label={`Ampliar ${label}`}><ExpandIcon /></button>
            </div>
        </figure>
    )
}

function imageExplanationKey(side, imageId) { return `${side}:${imageId}` }

function ComparisonImages({ taskType, leftItem, rightItem, leftEvaluation, rightEvaluation }) {
    const pairs = useMemo(() => getComparisonImagePairs(leftEvaluation, rightEvaluation), [leftEvaluation, rightEvaluation])
    const [visible, setVisible] = useState(8)
    const [filter, setFilter] = useState('all')
    const [preview, setPreview] = useState(null)
    const [explanations, setExplanations] = useState({})
    const sameModel = leftItem.model_id === rightItem.model_id
    const filteredPairs = useMemo(() => pairs.filter((pair) => matchesComparisonImageFilter(taskType, pair, filter)), [filter, pairs, taskType])
    const shown = filteredPairs.slice(0, visible)
    const explanationLabel = taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'
    const classificationSummary = useMemo(() => taskType === 'classification' ? getClassificationComparisonCaseSummary(leftEvaluation, rightEvaluation) : null, [leftEvaluation, rightEvaluation, taskType])
    const detectionImageSummary = useMemo(() => taskType === 'detection' ? getDetectionImageCaseSummary(leftEvaluation, rightEvaluation) : null, [leftEvaluation, rightEvaluation, taskType])

    async function requestExplanation(side, item, result) {
        const key = imageExplanationKey(side, result.image_id)
        setExplanations((current) => ({ ...current, [key]: { status: 'loading', data: null, error: '' } }))
        try {
            const data = taskType === 'detection'
                ? await generateDetectionEvaluationEigenCam(item.result_id, result.image_id)
                : await generateEvaluationGradCam(item.result_id, result.image_id)
            setExplanations((current) => ({ ...current, [key]: { status: 'success', data, error: '' } }))
        } catch (error) {
            const message = taskType === 'detection' ? getDetectionEigenCamErrorMessage(error) : getEvaluationGradCamErrorMessage(error)
            setExplanations((current) => ({ ...current, [key]: { status: 'error', data: null, error: message } }))
        }
    }

    const caseFilters = taskType === 'detection'
        ? [
            ['all', 'Todas', detectionImageSummary?.total ?? 0],
            ['left_more', 'Más detecciones en A', detectionImageSummary?.left_more ?? 0],
            ['right_more', 'Más detecciones en B', detectionImageSummary?.right_more ?? 0],
            ['different', 'Conteo diferente', detectionImageSummary?.different ?? 0],
            ['equal', 'Mismo conteo', detectionImageSummary?.equal ?? 0],
        ]
        : [
            ['all', 'Todas', classificationSummary?.total ?? 0],
            ['both_correct', 'Ambas aciertan', classificationSummary?.both_correct ?? 0],
            ['left_better', 'Solo A acierta', classificationSummary?.left_better ?? 0],
            ['right_better', 'Solo B acierta', classificationSummary?.right_better ?? 0],
            ['both_fail', 'Ambas fallan', classificationSummary?.both_fail ?? 0],
        ]

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Casos concretos" title="La misma imagen bajo A y B" description="Filtra casos divergentes y abre perturbaciones o explicabilidad solo cuando necesites profundizar." />
            <ComparisonGuide
                intro="Las miniaturas permiten detectar diferencias de un vistazo. Amplía cualquier vista para inspeccionar detalle sin perder bounding boxes ni etiquetas."
                items={[
                    { title: 'Filtros comparativos', text: taskType === 'detection' ? 'Priorizan imágenes donde A y B publican distinto número de detecciones.' : 'Priorizan casos donde una evaluación conserva la clase correcta y la otra no.' },
                    sameModel
                        ? { title: explanationLabel, text: `Puede solicitarse bajo demanda para cada evaluación sin repetir el ataque.` }
                        : { title: 'Explicabilidad entre modelos', text: `${explanationLabel} no se muestra al comparar modelos distintos porque su generación depende del modelo activo. Cambiarlo automáticamente alteraría el estado global de la aplicación.` },
                ]}
            />
            {classificationSummary && (
                <div className="evaluation-comparison-paired-stats" aria-label="Desacuerdo estadístico A/B">
                    <div><span>Tasa de desacuerdo</span><strong>{formatPercent(classificationSummary.disagreement_rate)}</strong><small>{classificationSummary.discordant} de {classificationSummary.total} imágenes tienen distinto resultado de acierto/fallo</small></div>
                    <div><span>McNemar exacto</span><strong>{classificationSummary.mcnemar_exact_p < 0.001 ? '< 0.001' : classificationSummary.mcnemar_exact_p.toFixed(3)}</strong><small>Contrasta si los desacuerdos están sistemáticamente inclinados hacia A o B</small></div>
                </div>
            )}
            <div className="evaluation-comparison-case-summary image-cases" role="group" aria-label="Filtros de casos A/B">
                {caseFilters.map(([id, label, count]) => (
                    <button key={id} type="button" className={`${filter === id ? 'active ' : ''}${id === 'left_better' || id === 'left_more' ? 'side-a' : id === 'right_better' || id === 'right_more' ? 'side-b' : ''}`.trim()} onClick={() => { setFilter(id); setVisible(8) }}>
                        <strong>{count}</strong><span>{label}</span>
                    </button>
                ))}
            </div>
            {classificationSummary && <p className="evaluation-comparison-stat-note"><strong>Cómo leer McNemar:</strong> un p pequeño aporta evidencia de que los desacuerdos favorecen sistemáticamente a una evaluación en estas mismas imágenes. No mide el tamaño ni la importancia práctica de la diferencia.</p>}
            {shown.map(({ left, right }) => {
                const leftExplanation = explanations[imageExplanationKey('a', left.image_id)] ?? { status: 'idle' }
                const rightExplanation = explanations[imageExplanationKey('b', right.image_id)] ?? { status: 'idle' }
                return (
                    <article key={left.image_id} className="evaluation-comparison-image-case">
                        <header><div><strong>{left.filename ?? left.image_id}</strong><small>{left.image_id}</small></div><div className="evaluation-comparison-image-outcomes"><span className="side-a">A · {taskType === 'detection' ? `${left.adversarial_detection_count} detecciones` : `${left.adversarial_label} · ${formatPercent(left.adversarial_confidence)}`}</span><span className="side-b">B · {taskType === 'detection' ? `${right.adversarial_detection_count} detecciones` : `${right.adversarial_label} · ${formatPercent(right.adversarial_confidence)}`}</span></div></header>
                        <div className={`evaluation-comparison-image-grid${taskType === 'detection' ? ' detection' : ''}`}>
                            {taskType === 'detection' ? <>
                                <ComparisonImageFigure label="A · Original" url={`/datasets/${leftEvaluation.dataset_id}/images/${left.image_id}`} caption={`${left.original_detection_count} cajas`} detections={left.original_detections} onExpand={setPreview} tone="side-a" />
                                <ComparisonImageFigure label="A · Tras el ataque" url={left.adversarial_image_url} caption={`${left.adversarial_detection_count} cajas`} detections={left.adversarial_detections} onExpand={setPreview} tone="side-a" />
                                <ComparisonImageFigure label="B · Original" url={`/datasets/${rightEvaluation.dataset_id}/images/${right.image_id}`} caption={`${right.original_detection_count} cajas`} detections={right.original_detections} onExpand={setPreview} tone="side-b" />
                                <ComparisonImageFigure label="B · Tras el ataque" url={right.adversarial_image_url} caption={`${right.adversarial_detection_count} cajas`} detections={right.adversarial_detections} onExpand={setPreview} tone="side-b" />
                            </> : <>
                                <ComparisonImageFigure label="Original" url={left.original_image_url} caption={left.ground_truth_label} onExpand={setPreview} tone="reference" />
                                <ComparisonImageFigure label="Evaluación A" url={left.adversarial_image_url} caption={`${left.original_label} → ${left.adversarial_label} · ${formatPercent(left.original_confidence)} → ${formatPercent(left.adversarial_confidence)}`} onExpand={setPreview} tone="side-a" />
                                <ComparisonImageFigure label="Evaluación B" url={right.adversarial_image_url} caption={`${right.original_label} → ${right.adversarial_label} · ${formatPercent(right.original_confidence)} → ${formatPercent(right.adversarial_confidence)}`} onExpand={setPreview} tone="side-b" />
                            </>}
                        </div>
                        <details className="evaluation-comparison-image-extra">
                            <summary>Perturbación <span>comparar magnitud{taskType === 'classification' ? ' y mapa' : ''}</span></summary>
                            <div className={`evaluation-comparison-perturbation-grid${taskType === 'classification' ? '' : ' metrics-only'}`}>
                                {taskType === 'classification' && <ComparisonImageFigure label="Perturbación A" url={left.perturbation_image_url} caption={`L∞ ${formatNumber(left.perturbation_linf, 4)} · L2 ${formatNumber(left.perturbation_l2, 4)}`} onExpand={setPreview} tone="side-a" />}
                                {taskType === 'classification' && <ComparisonImageFigure label="Perturbación B" url={right.perturbation_image_url} caption={`L∞ ${formatNumber(right.perturbation_linf, 4)} · L2 ${formatNumber(right.perturbation_l2, 4)}`} onExpand={setPreview} tone="side-b" />}
                                {taskType === 'detection' && <><div className="side-a"><strong>A</strong><span>L∞ {formatNumber(left.perturbation_linf, 4)}</span><span>L2 {formatNumber(left.perturbation_l2, 4)}</span></div><div className="side-b"><strong>B</strong><span>L∞ {formatNumber(right.perturbation_linf, 4)}</span><span>L2 {formatNumber(right.perturbation_l2, 4)}</span></div></>}
                            </div>
                        </details>
                        {sameModel && <details className="evaluation-comparison-image-extra">
                            <summary>{explanationLabel} <span>explicabilidad bajo demanda</span></summary>
                            <div className="evaluation-comparison-explain-actions">
                                <button type="button" disabled={leftExplanation.status === 'loading' || leftExplanation.status === 'success'} onClick={() => requestExplanation('a', leftItem, left)}>{leftExplanation.status === 'loading' ? `Generando ${explanationLabel} A…` : leftExplanation.status === 'success' ? `${explanationLabel} A generado` : `Generar ${explanationLabel} A`}</button>
                                <button type="button" disabled={rightExplanation.status === 'loading' || rightExplanation.status === 'success'} onClick={() => requestExplanation('b', rightItem, right)}>{rightExplanation.status === 'loading' ? `Generando ${explanationLabel} B…` : rightExplanation.status === 'success' ? `${explanationLabel} B generado` : `Generar ${explanationLabel} B`}</button>
                            </div>
                            {(leftExplanation.error || rightExplanation.error) && <div className="evaluation-comparison-explain-errors">{leftExplanation.error && <p><strong>A:</strong> {leftExplanation.error}</p>}{rightExplanation.error && <p><strong>B:</strong> {rightExplanation.error}</p>}</div>}
                            <div className="evaluation-comparison-explain-grid">
                                {leftExplanation.status === 'success' && (taskType === 'detection' ? <><ComparisonImageFigure label={`${explanationLabel} A · original`} url={leftExplanation.data.original_eigencam_image_url} onExpand={setPreview} tone="side-a" /><ComparisonImageFigure label={`${explanationLabel} A · ataque`} url={leftExplanation.data.adversarial_eigencam_image_url} onExpand={setPreview} tone="side-a" /></> : <><ComparisonImageFigure label={`${explanationLabel} A · original`} url={leftExplanation.data.original_gradcam_image_url} onExpand={setPreview} tone="side-a" /><ComparisonImageFigure label={`${explanationLabel} A · ataque`} url={leftExplanation.data.adversarial_gradcam_image_url} onExpand={setPreview} tone="side-a" /></>)}
                                {rightExplanation.status === 'success' && (taskType === 'detection' ? <><ComparisonImageFigure label={`${explanationLabel} B · original`} url={rightExplanation.data.original_eigencam_image_url} onExpand={setPreview} tone="side-b" /><ComparisonImageFigure label={`${explanationLabel} B · ataque`} url={rightExplanation.data.adversarial_eigencam_image_url} onExpand={setPreview} tone="side-b" /></> : <><ComparisonImageFigure label={`${explanationLabel} B · original`} url={rightExplanation.data.original_gradcam_image_url} onExpand={setPreview} tone="side-b" /><ComparisonImageFigure label={`${explanationLabel} B · ataque`} url={rightExplanation.data.adversarial_gradcam_image_url} onExpand={setPreview} tone="side-b" /></>)}
                            </div>
                        </details>}
                    </article>
                )
            })}
            {!shown.length && <EmptyComparison title="No hay casos para este filtro">Selecciona otro criterio para explorar las imágenes compartidas por A y B.</EmptyComparison>}
            {filteredPairs.length > 8 && <div className="evaluation-comparison-more"><span>{Math.min(visible, filteredPairs.length)} / {filteredPairs.length}</span>{visible < filteredPairs.length ? <button type="button" onClick={() => setVisible((current) => Math.min(filteredPairs.length, current + 8))}>Ver 8 más</button> : <button type="button" onClick={() => setVisible(8)}>Mostrar menos</button>}</div>}
            {preview && <ComparisonImageLightbox preview={preview} onClose={() => setPreview(null)} />}
        </section>
    )
}

function DetectionCases({ leftItem, rightItem, leftEvaluation, rightEvaluation }) {
    const [mode, setMode] = useState('objects')
    return (
        <section className="evaluation-comparison-cases-shell">
            <div className="evaluation-comparison-view-switcher" role="group" aria-label="Tipo de caso clave">
                <button type="button" className={mode === 'objects' ? 'active' : ''} onClick={() => setMode('objects')}>Objetos anotados</button>
                <button type="button" className={mode === 'images' ? 'active' : ''} onClick={() => setMode('images')}>Imágenes completas</button>
            </div>
            {mode === 'objects'
                ? <DetectionObjectCases leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />
                : <ComparisonImages taskType="detection" leftItem={leftItem} rightItem={rightItem} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
        </section>
    )
}

function ComparisonDiagnostics({ taskType, leftEvaluation, rightEvaluation, sameModel }) {
    const [mode, setMode] = useState(taskType === 'detection' ? 'coco' : 'curves')
    if (taskType === 'classification') {
        return <ClassificationCurveComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} sameModel={sameModel} />
    }
    return (
        <section className="evaluation-comparison-diagnostics-shell">
            <div className="evaluation-comparison-view-switcher" role="group" aria-label="Herramienta de diagnóstico">
                <button type="button" className={mode === 'coco' ? 'active' : ''} onClick={() => setMode('coco')}>COCO</button>
                <button type="button" className={mode === 'curves' ? 'active' : ''} onClick={() => setMode('curves')}>Precision–Recall</button>
                <button type="button" className={mode === 'tide' ? 'active' : ''} onClick={() => setMode('tide')}>TIDE</button>
            </div>
            {mode === 'coco' && <DetectionCocoComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
            {mode === 'curves' && <DetectionCurveComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
            {mode === 'tide' && <DetectionTideComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
        </section>
    )
}

export function EvaluationComparisonDetails({ taskType, leftItem, rightItem, leftEvaluation, rightEvaluation, summary }) {
    const tabs = [
        { id: 'summary', label: 'Resumen' },
        { id: 'differences', label: 'Diferencias' },
        { id: 'cases', label: 'Casos clave' },
        { id: 'diagnosis', label: 'Diagnóstico' },
    ]
    const [activeTab, setActiveTab] = useState('summary')
    const sameModel = leftItem.model_id === rightItem.model_id

    return (
        <div className="evaluation-comparison-details redesigned">
            <ComparisonTabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
            <div className="evaluation-comparison-detail-content">
                {activeTab === 'summary' && summary}
                {activeTab === 'differences' && taskType === 'classification' && <ClassificationClassComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {activeTab === 'differences' && taskType === 'detection' && <DetectionClassComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {activeTab === 'cases' && taskType === 'classification' && <ComparisonImages taskType="classification" leftItem={leftItem} rightItem={rightItem} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {activeTab === 'cases' && taskType === 'detection' && <DetectionCases leftItem={leftItem} rightItem={rightItem} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {activeTab === 'diagnosis' && <ComparisonDiagnostics taskType={taskType} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} sameModel={sameModel} />}
            </div>
        </div>
    )
}

export default EvaluationComparisonDetails
