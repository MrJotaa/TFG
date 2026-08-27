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
    getComparisonImagePairs,
    getDetectionClassComparison,
    getDetectionCocoComparisonRows,
    getDetectionObjectComparisonPairs,
    getDetectionTideComparisonRows,
    matchesComparisonImageFilter,
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

function ComparisonCurveTooltip({ active, payload, xLabel = 'X' }) {
    if (!active || !payload?.length) return null
    const point = payload[0]?.payload
    return (
        <div className="evaluation-chart-tooltip comparison-chart-tooltip">
            <strong>{xLabel}: {formatPercent(point?.x ?? point?.recall, 1)}</strong>
            {payload.map((entry) => (
                <div key={`${entry.name}-${entry.dataKey}`}>
                    <span style={{ '--series-color': entry.color }}>{entry.name}</span>
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

function ComparisonCurveChart({ series, yLabel = 'Valor', xLabel = 'Umbral', domain = [0, 1] }) {
    const normalizedSeries = series
        .map((entry) => ({ ...entry, points: normalizeComparisonCurve(entry.points, entry.xKey ?? 'x') }))
        .filter((entry) => entry.points.length > 0)

    if (!normalizedSeries.length) {
        return <EmptyComparison title="Curva no disponible">Las evaluaciones no contienen una curva representable para este ámbito.</EmptyComparison>
    }

    return (
        <div className="evaluation-comparison-chart" role="img" aria-label="Comparación de curvas A y B">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart margin={{ top: 12, right: 20, bottom: 18, left: 10 }}>
                    <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="4 4" />
                    <XAxis
                        dataKey="x"
                        type="number"
                        domain={domain}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        tick={{ fill: 'var(--muted)', fontSize: 11 }}
                        label={{ value: xLabel, position: 'insideBottom', offset: -10, fill: 'var(--muted)' }}
                    />
                    <YAxis
                        type="number"
                        domain={domain}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        tick={{ fill: 'var(--muted)', fontSize: 11 }}
                        label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: 'var(--muted)' }}
                    />
                    <Tooltip content={<ComparisonCurveTooltip xLabel={xLabel} />} />
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
                            activeDot={{ r: 4 }}
                            isAnimationActive={false}
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>
        </div>
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
    return evaluation?.advanced_metrics?.supervised?.[curveType]?.micro?.curves
}

function ClassificationCurveComparison({ leftEvaluation, rightEvaluation, sameModel }) {
    const [curveType, setCurveType] = useState('roc')
    const left = getClassificationCurve(leftEvaluation, curveType)
    const right = getClassificationCurve(rightEvaluation, curveType)
    const series = [
        ...(sameModel && left?.original?.available ? [{ key: 'reference', label: 'Referencia original común', color: 'var(--comparison-reference)', reference: true, points: left.original.points }] : []),
        ...(!sameModel && left?.original?.available ? [{ key: 'a-original', label: 'A · original', color: 'var(--comparison-a)', reference: true, points: left.original.points }] : []),
        ...(!sameModel && right?.original?.available ? [{ key: 'b-original', label: 'B · original', color: 'var(--comparison-b)', reference: true, points: right.original.points }] : []),
        ...(left?.adversarial?.available ? [{ key: 'a', label: 'A · robusta', color: 'var(--comparison-a)', points: left.adversarial.points }] : []),
        ...(right?.adversarial?.available ? [{ key: 'b', label: 'B · robusta', color: 'var(--comparison-b)', points: right.adversarial.points }] : []),
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
                ]}
            />
            <ComparisonCurveChart series={series} xLabel={curveType === 'roc' ? 'Falsos positivos' : 'Recall'} yLabel={curveType === 'roc' ? 'Verdaderos positivos' : 'Precisión'} />
        </section>
    )
}

function DetectionCocoComparison({ leftEvaluation, rightEvaluation }) {
    const rows = getDetectionCocoComparisonRows(leftEvaluation, rightEvaluation)
    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Protocolo COCO" title="Métricas robustas A frente a B" description="La comparación utiliza exactamente el mismo dataset y protocolo supervisado; el delta se calcula como B − A." />
            <ComparisonGuide
                intro="Usa la tabla para cuantificar diferencias exactas entre las dos evaluaciones bajo el mismo protocolo COCO."
                items={[
                    { title: 'AP', text: 'Resume precisión a lo largo de la curva Precision–Recall. AP@0.50 es más permisivo y AP@0.75 exige una localización más precisa.' },
                    { title: 'AR', text: 'Mide qué fracción de los objetos de referencia puede recuperarse permitiendo un máximo de detecciones por imagen.' },
                ]}
                note="El delta es B − A. En AP y AR un valor mayor representa mejor rendimiento."
            />
            <ComparisonDataTable
                columns={[
                    { key: 'label', label: 'Métrica', width: '1.6fr', render: (row) => <strong>{row.label}</strong> },
                    { key: 'left', label: 'A', className: 'side-a', render: (row) => formatPercent(row.left) },
                    { key: 'right', label: 'B', className: 'side-b', render: (row) => formatPercent(row.right) },
                    { key: 'delta', label: 'Δ B − A', render: (row) => <strong className={(row.delta ?? 0) > 0 ? 'tone-success' : (row.delta ?? 0) < 0 ? 'tone-danger' : ''}>{formatPoints(row.delta)}</strong> },
                ]}
                rows={rows}
            />
        </section>
    )
}

function DetectionClassComparison({ leftEvaluation, rightEvaluation }) {
    const [query, setQuery] = useState('')
    const [sort, setSort] = useState('difference')
    const rows = useMemo(() => {
        const normalized = query.trim().toLocaleLowerCase('es')
        return getDetectionClassComparison(leftEvaluation, rightEvaluation)
            .filter((row) => !normalized || row.label.toLocaleLowerCase('es').includes(normalized))
            .sort((left, right) => sort === 'support'
                ? right.support - left.support
                : sort === 'name'
                    ? left.label.localeCompare(right.label, 'es')
                    : Math.abs(right.delta ?? -Infinity) - Math.abs(left.delta ?? -Infinity))
    }, [leftEvaluation, query, rightEvaluation, sort])

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Desglose por categoría" title="AP, AR y supresión por clase" description="Las barras hacen visible la diferencia de AP; debajo se conservan AR y objetos suprimidos como contexto numérico." controls={<label><span>Orden</span><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="difference">Mayor diferencia AP</option><option value="support">Mayor soporte</option><option value="name">Nombre</option></select></label>} />
            <ComparisonGuide intro="Una diferencia grande de AP puede tener causas distintas. Contrástala con AR, supresión y soporte antes de interpretarla." items={[{ title: 'AP', text: 'Resume la calidad Precision–Recall de la categoría; cuanto mayor, mejor.' }, { title: 'Supresión', text: 'Cuenta objetos originalmente detectados que dejan de detectarse tras el ataque.' }]} />
            <div className="evaluation-comparison-search-row"><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar categoría" /><span>{rows.length} clases</span></div>
            <div className="evaluation-comparison-class-list">
                {rows.map((row) => (
                    <article key={row.key} className="evaluation-comparison-class-row detection">
                        <header><div><strong>{row.label}</strong><small>{row.support} objetos</small></div><span>Δ AP {formatPoints(row.delta)}</span></header>
                        <div className="evaluation-comparison-class-bars">
                            <div><span className="side-a">A</span><i><b className="side-a" style={{ width: `${Math.max(0, Math.min(1, row.leftAp ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row.leftAp)}</strong></div>
                            <div><span className="side-b">B</span><i><b className="side-b" style={{ width: `${Math.max(0, Math.min(1, row.rightAp ?? 0)) * 100}%` }} /></i><strong>{formatPercent(row.rightAp)}</strong></div>
                        </div>
                        <dl className="evaluation-comparison-class-meta"><div><dt>AR A / B</dt><dd>{formatPercent(row.leftAr)} / {formatPercent(row.rightAr)}</dd></div><div><dt>Suprimidos A / B</dt><dd>{row.leftSuppressed} / {row.rightSuppressed}</dd></div></dl>
                    </article>
                ))}
            </div>
        </section>
    )
}

function getDetectionPrSeries(evaluation, scope) {
    const supervised = evaluation?.supervised
    if (!supervised?.available) return null
    if (scope === 'global') return supervised.adversarial?.precision_recall_curve
    return supervised.adversarial?.per_class?.find((entry) => String(entry.reference?.category_internal_id ?? entry.reference?.category_name) === scope)?.precision_recall_curve
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
                    { key: 'a', label: 'Evaluación A', color: 'var(--comparison-a)', points: leftCurve?.points, xKey: 'recall' },
                    { key: 'b', label: 'Evaluación B', color: 'var(--comparison-b)', points: rightCurve?.points, xKey: 'recall' },
                ]}
                xLabel="Recall"
                yLabel="Precisión"
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
                        <Tooltip cursor={false} content={<ComparisonTideTooltip />} />
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

function ObjectObservation({ object }) {
    if (!object) return '—'
    const observation = object.adversarial_observation
    if (!observation) return 'No detectado'
    return `${formatNumber(observation.confidence)} conf. · IoU ${formatNumber(observation.iou_with_ground_truth)}`
}

function formatBox(box) {
    if (!box) return '—'
    return `[${formatNumber(box.x1, 1)}, ${formatNumber(box.y1, 1)}, ${formatNumber(box.x2, 1)}, ${formatNumber(box.y2, 1)}]`
}

function DetectionObjectComparison({ leftEvaluation, rightEvaluation }) {
    const pairs = useMemo(() => getDetectionObjectComparisonPairs(leftEvaluation, rightEvaluation), [leftEvaluation, rightEvaluation])
    const [query, setQuery] = useState('')
    const [page, setPage] = useState(1)
    const [expandedId, setExpandedId] = useState(null)
    const pageSize = 25
    const normalizedQuery = query.trim().toLocaleLowerCase('es')
    const filtered = pairs.filter(({ left }) => !normalizedQuery || [left.object_id, left.image_id, left.category?.category_name].some((value) => String(value ?? '').toLocaleLowerCase('es').includes(normalizedQuery)))
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
    const safePage = Math.min(page, totalPages)
    const pageItems = filtered.slice((safePage - 1) * pageSize, safePage * pageSize)

    if (!pairs.length) return <EmptyComparison title="Detalle por objeto no disponible">Las dos evaluaciones deben incluir el detalle materializado por objeto.</EmptyComparison>

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Objetos evaluados" title="Transición de cada objeto en A y B" description="Abre cualquier fila para inspeccionar observaciones, cajas y deltas del mismo objeto anotado en las dos evaluaciones." />
            <ComparisonGuide intro="La fila resume el desenlace. El detalle permite comprobar si la diferencia procede de confianza, solapamiento IoU o desaparición de la detección." items={[{ title: 'Preservado', text: 'El objeto se detecta antes y después del ataque.' }, { title: 'Suprimido / recuperado', text: 'Indica que el objeto aparece solo antes o solo después del ataque, respectivamente.' }]} />
            <div className="evaluation-comparison-search-row"><input type="search" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); setExpandedId(null) }} placeholder="Buscar objeto, imagen o clase" /><span>{filtered.length} objetos</span></div>
            <div className="evaluation-comparison-object-list">
                {pageItems.map((row) => {
                    const expanded = expandedId === row.left.object_id
                    return (
                        <article key={row.left.object_id} className={`evaluation-comparison-object-row${expanded ? ' expanded' : ''}`}>
                            <button type="button" className="evaluation-comparison-object-summary" aria-expanded={expanded} onClick={() => setExpandedId(expanded ? null : row.left.object_id)}>
                                <span><strong>{row.left.category?.category_name ?? 'Objeto'}</strong><small>{row.left.object_id} · imagen {row.left.image_id}</small></span>
                                <span className="side-a"><small>A</small><strong>{row.left.outcome}</strong><em>{ObjectObservation({ object: row.left })}</em></span>
                                <span className="side-b"><small>B</small><strong>{row.right.outcome}</strong><em>{ObjectObservation({ object: row.right })}</em></span>
                                <span className="evaluation-comparison-object-chevron" aria-hidden="true">⌄</span>
                            </button>
                            {expanded && (
                                <div className="evaluation-comparison-object-detail">
                                    <dl><div><dt>Clase</dt><dd>{row.left.category?.category_name}</dd></div><div><dt>Tamaño COCO</dt><dd>{row.left.size_bucket}</dd></div><div><dt>Área</dt><dd>{formatNumber(row.left.area, 1)} px²</dd></div><div><dt>Caja de referencia</dt><dd>{formatBox(row.left.ground_truth_box)}</dd></div></dl>
                                    <section className="side-a"><strong>Evaluación A</strong><dl><div><dt>Caja original</dt><dd>{formatBox(row.left.original_observation?.detection_box)}</dd></div><div><dt>Caja tras ataque</dt><dd>{formatBox(row.left.adversarial_observation?.detection_box)}</dd></div><div><dt>Δ confianza</dt><dd>{formatNumber(row.left.confidence_drop)}</dd></div><div><dt>Δ IoU</dt><dd>{formatNumber(row.left.iou_drop)}</dd></div></dl></section>
                                    <section className="side-b"><strong>Evaluación B</strong><dl><div><dt>Caja original</dt><dd>{formatBox(row.right.original_observation?.detection_box)}</dd></div><div><dt>Caja tras ataque</dt><dd>{formatBox(row.right.adversarial_observation?.detection_box)}</dd></div><div><dt>Δ confianza</dt><dd>{formatNumber(row.right.confidence_drop)}</dd></div><div><dt>Δ IoU</dt><dd>{formatNumber(row.right.iou_drop)}</dd></div></dl></section>
                                </div>
                            )}
                        </article>
                    )
                })}
            </div>
            {totalPages > 1 && <div className="evaluation-comparison-pagination"><button type="button" disabled={safePage === 1} onClick={() => { setPage((current) => Math.max(1, current - 1)); setExpandedId(null) }}>Anterior</button><span>Página {safePage} de {totalPages}</span><button type="button" disabled={safePage === totalPages} onClick={() => { setPage((current) => Math.min(totalPages, current + 1)); setExpandedId(null) }}>Siguiente</button></div>}
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
                const label = `${detection.label ?? 'objeto'} · ${formatNumber(detection.confidence, 2)}`
                const labelWidth = Math.min(box.x2 - box.x1, Math.max(fontSize * 4.5, label.length * fontSize * 0.58))
                const labelY = Math.max(0, box.y1 - fontSize * 1.45)
                return (
                    <g key={`${detection.label ?? 'detection'}-${index}`}>
                        <rect className="evaluation-comparison-detection-box" x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1} strokeWidth={strokeWidth} />
                        <rect className="evaluation-comparison-detection-label-bg" x={box.x1} y={labelY} width={labelWidth} height={fontSize * 1.45} />
                        <text className="evaluation-comparison-detection-label" x={box.x1 + fontSize * 0.25} y={labelY + fontSize} fontSize={fontSize}>{label}</text>
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

    const filters = taskType === 'detection'
        ? [['all', 'Todas'], ['left_more', 'Más detecciones en A'], ['right_more', 'Más detecciones en B'], ['different', 'Conteo diferente'], ['equal', 'Mismo conteo']]
        : [['all', 'Todas'], ['left_better', 'A acierta / B falla'], ['right_better', 'B acierta / A falla'], ['both_fail', 'Ambas fallan'], ['both_correct', 'Ambas aciertan']]

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Casos concretos" title="La misma imagen bajo A y B" description="Filtra casos divergentes y abre perturbaciones o explicabilidad solo cuando necesites profundizar." />
            <ComparisonGuide intro="Las miniaturas permiten detectar diferencias de un vistazo. Amplía cualquier vista para inspeccionar detalle sin perder bounding boxes ni etiquetas." items={[{ title: 'Filtros comparativos', text: taskType === 'detection' ? 'Priorizan imágenes donde A y B publican distinto número de detecciones.' : 'Priorizan casos donde una evaluación conserva la clase correcta y la otra no.' }, { title: explanationLabel, text: sameModel ? `Puede solicitarse bajo demanda para cada evaluación sin repetir el ataque.` : `La generación bajo demanda se reserva a comparaciones con el mismo modelo para no cambiar el modelo activo global.` }]} />
            <div className="evaluation-comparison-image-filters">{filters.map(([id, label]) => <button key={id} type="button" className={filter === id ? 'active' : ''} onClick={() => { setFilter(id); setVisible(8) }}>{label}</button>)}<span>{filteredPairs.length} casos</span></div>
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
                        <details className="evaluation-comparison-image-extra">
                            <summary>{explanationLabel} <span>explicabilidad bajo demanda</span></summary>
                            {!sameModel ? <p className="evaluation-comparison-explain-note">Para comparar modelos distintos no cambiamos automáticamente el modelo activo. Así evitamos alterar el estado global de la aplicación durante una comparación histórica.</p> : <>
                                <div className="evaluation-comparison-explain-actions">
                                    <button type="button" disabled={leftExplanation.status === 'loading' || leftExplanation.status === 'success'} onClick={() => requestExplanation('a', leftItem, left)}>{leftExplanation.status === 'loading' ? `Generando ${explanationLabel} A…` : leftExplanation.status === 'success' ? `${explanationLabel} A generado` : `Generar ${explanationLabel} A`}</button>
                                    <button type="button" disabled={rightExplanation.status === 'loading' || rightExplanation.status === 'success'} onClick={() => requestExplanation('b', rightItem, right)}>{rightExplanation.status === 'loading' ? `Generando ${explanationLabel} B…` : rightExplanation.status === 'success' ? `${explanationLabel} B generado` : `Generar ${explanationLabel} B`}</button>
                                </div>
                                {(leftExplanation.error || rightExplanation.error) && <div className="evaluation-comparison-explain-errors">{leftExplanation.error && <p><strong>A:</strong> {leftExplanation.error}</p>}{rightExplanation.error && <p><strong>B:</strong> {rightExplanation.error}</p>}</div>}
                                <div className="evaluation-comparison-explain-grid">
                                    {leftExplanation.status === 'success' && (taskType === 'detection' ? <><ComparisonImageFigure label={`${explanationLabel} A · original`} url={leftExplanation.data.original_eigencam_image_url} onExpand={setPreview} tone="side-a" /><ComparisonImageFigure label={`${explanationLabel} A · ataque`} url={leftExplanation.data.adversarial_eigencam_image_url} onExpand={setPreview} tone="side-a" /></> : <><ComparisonImageFigure label={`${explanationLabel} A · original`} url={leftExplanation.data.original_gradcam_image_url} onExpand={setPreview} tone="side-a" /><ComparisonImageFigure label={`${explanationLabel} A · ataque`} url={leftExplanation.data.adversarial_gradcam_image_url} onExpand={setPreview} tone="side-a" /></>)}
                                    {rightExplanation.status === 'success' && (taskType === 'detection' ? <><ComparisonImageFigure label={`${explanationLabel} B · original`} url={rightExplanation.data.original_eigencam_image_url} onExpand={setPreview} tone="side-b" /><ComparisonImageFigure label={`${explanationLabel} B · ataque`} url={rightExplanation.data.adversarial_eigencam_image_url} onExpand={setPreview} tone="side-b" /></> : <><ComparisonImageFigure label={`${explanationLabel} B · original`} url={rightExplanation.data.original_gradcam_image_url} onExpand={setPreview} tone="side-b" /><ComparisonImageFigure label={`${explanationLabel} B · ataque`} url={rightExplanation.data.adversarial_gradcam_image_url} onExpand={setPreview} tone="side-b" /></>)}
                                </div>
                            </>}
                        </details>
                    </article>
                )
            })}
            {!shown.length && <EmptyComparison title="No hay casos para este filtro">Selecciona otro criterio para explorar las imágenes compartidas por A y B.</EmptyComparison>}
            {filteredPairs.length > 8 && <div className="evaluation-comparison-more"><span>{Math.min(visible, filteredPairs.length)} / {filteredPairs.length}</span>{visible < filteredPairs.length ? <button type="button" onClick={() => setVisible((current) => Math.min(filteredPairs.length, current + 8))}>Ver 8 más</button> : <button type="button" onClick={() => setVisible(8)}>Mostrar menos</button>}</div>}
            {preview && <ComparisonImageLightbox preview={preview} onClose={() => setPreview(null)} />}
        </section>
    )
}

export function EvaluationComparisonDetails({ taskType, leftItem, rightItem, leftEvaluation, rightEvaluation, summary }) {
    const tabs = taskType === 'detection'
        ? [
            { id: 'summary', label: 'Resumen' },
            { id: 'coco', label: 'Métricas COCO' },
            { id: 'classes', label: 'Por clase', badge: leftEvaluation?.supervised?.original?.per_class?.length },
            { id: 'curves', label: 'Curvas PR' },
            { id: 'objects', label: 'Por objeto', badge: leftEvaluation?.supervised?.object_analysis?.objects?.length },
            { id: 'tide', label: 'TIDE' },
            { id: 'images', label: 'Imágenes', badge: leftEvaluation?.results?.length },
        ]
        : [
            { id: 'summary', label: 'Resumen' },
            { id: 'classes', label: 'Por clase', badge: leftEvaluation?.advanced_metrics?.supervised?.per_class?.length },
            { id: 'curves', label: 'Curvas' },
            { id: 'images', label: 'Imágenes', badge: leftEvaluation?.results?.length },
        ]
    const [activeTab, setActiveTab] = useState('summary')
    const sameModel = leftItem.model_id === rightItem.model_id

    return (
        <div className="evaluation-comparison-details">
            <ComparisonTabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
            <div className="evaluation-comparison-detail-content">
                {activeTab === 'summary' && summary}
                {taskType === 'classification' && activeTab === 'classes' && <ClassificationClassComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'classification' && activeTab === 'curves' && <ClassificationCurveComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} sameModel={sameModel} />}
                {taskType === 'classification' && activeTab === 'images' && <ComparisonImages taskType={taskType} leftItem={leftItem} rightItem={rightItem} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'coco' && <DetectionCocoComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'classes' && <DetectionClassComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'curves' && <DetectionCurveComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'objects' && <DetectionObjectComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'tide' && <DetectionTideComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'images' && <ComparisonImages taskType={taskType} leftItem={leftItem} rightItem={rightItem} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
            </div>
        </div>
    )
}

export default EvaluationComparisonDetails
