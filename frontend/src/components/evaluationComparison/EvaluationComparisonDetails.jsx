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
import {
    getClassificationClassComparison,
    getComparisonImagePairs,
    getDetectionClassComparison,
    getDetectionCocoComparisonRows,
    getDetectionObjectComparisonPairs,
    getDetectionTideComparisonRows,
} from '../../utils/evaluationComparison'
import {
    clampEvaluationPreviewPan,
    clampEvaluationPreviewZoom,
    getEvaluationPreviewPanBounds,
    stepEvaluationPreviewZoom,
} from '../../utils/evaluationImagePreview'
import { TIDE_ERROR_DEFINITIONS } from '../../utils/detectionMetrics'

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
                <div key={entry.dataKey}>
                    <span style={{ '--series-color': entry.color }}>{entry.name}</span>
                    <strong>{formatPercent(entry.value, 1)}</strong>
                </div>
            ))}
        </div>
    )
}

function ComparisonCurveChart({ series, xKey = 'x', yLabel = 'Valor', xLabel = 'Umbral', domain = [0, 1] }) {
    const merged = useMemo(() => {
        const points = new Map()
        for (const entry of series) {
            for (const point of entry.points ?? []) {
                const x = Number(point[xKey])
                const y = Number(point.y ?? point.precision)
                if (!Number.isFinite(x) || !Number.isFinite(y)) continue
                points.set(x, { ...(points.get(x) ?? {}), [xKey]: x, [entry.key]: y })
            }
        }
        return [...points.values()].sort((left, right) => Number(left[xKey]) - Number(right[xKey]))
    }, [series, xKey])

    if (!merged.length) {
        return <EmptyComparison title="Curva no disponible">No hay puntos suficientes para representar esta comparación.</EmptyComparison>
    }

    return (
        <div className="evaluation-comparison-chart" role="img" aria-label="Comparación de curvas A y B">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={merged} margin={{ top: 12, right: 20, bottom: 18, left: 10 }}>
                    <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="4 4" />
                    <XAxis
                        dataKey={xKey}
                        type="number"
                        domain={domain}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        tick={{ fill: 'var(--muted)', fontSize: 11 }}
                        label={{ value: xLabel, position: 'insideBottom', offset: -10, fill: 'var(--muted)' }}
                    />
                    <YAxis
                        domain={domain}
                        tickFormatter={(value) => Number(value).toFixed(2)}
                        tick={{ fill: 'var(--muted)', fontSize: 11 }}
                        label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: 'var(--muted)' }}
                    />
                    <Tooltip content={<ComparisonCurveTooltip xLabel={xLabel} />} />
                    {series.map((entry) => (
                        <Line
                            key={entry.key}
                            dataKey={entry.key}
                            name={entry.label}
                            stroke={entry.color}
                            strokeWidth={entry.reference ? 2 : 3}
                            strokeDasharray={entry.reference ? '7 5' : undefined}
                            dot={false}
                            connectNulls
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
                description="Compara el rendimiento robusto de la misma clase en las dos evaluaciones y ordena por las diferencias más relevantes."
                controls={(
                    <>
                        <label><span>Métrica</span><select value={metric} onChange={(event) => setMetric(event.target.value)}>{CLASSIFICATION_METRICS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
                        <label><span>Orden</span><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="difference">Mayor diferencia</option><option value="support">Mayor soporte</option><option value="name">Nombre</option></select></label>
                    </>
                )}
            />
            <div className="evaluation-comparison-search-row">
                <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar clase" />
                <span>{rows.length} clases</span>
            </div>
            <ComparisonDataTable
                columns={[
                    { key: 'label', label: 'Clase', width: '1.6fr', render: (row) => <><strong>{row.label}</strong><small>Soporte {row.support}</small></> },
                    { key: 'leftRobust', label: 'A robusto', className: 'side-a', render: (row) => formatPercent(row.leftRobust) },
                    { key: 'rightRobust', label: 'B robusto', className: 'side-b', render: (row) => formatPercent(row.rightRobust) },
                    { key: 'delta', label: 'Δ B − A', render: (row) => <strong className={(row.delta ?? 0) > 0 ? 'tone-success' : (row.delta ?? 0) < 0 ? 'tone-danger' : ''}>{formatPoints(row.delta)}</strong> },
                ]}
                rows={rows}
            />
        </section>
    )
}

function getClassificationCurve(evaluation, curveType) {
    return evaluation?.advanced_metrics?.supervised?.[curveType]?.micro?.curves
}

function ClassificationCurveComparison({ leftEvaluation, rightEvaluation, showReference }) {
    const [curveType, setCurveType] = useState('roc')
    const left = getClassificationCurve(leftEvaluation, curveType)
    const right = getClassificationCurve(rightEvaluation, curveType)
    const series = [
        ...(showReference && left?.original?.available ? [{ key: 'reference', label: 'Referencia original', color: 'var(--evaluation-original)', reference: true, points: left.original.points }] : []),
        ...(left?.adversarial?.available ? [{ key: 'a', label: 'Evaluación A', color: 'var(--comparison-a)', points: left.adversarial.points }] : []),
        ...(right?.adversarial?.available ? [{ key: 'b', label: 'Evaluación B', color: 'var(--comparison-b)', points: right.adversarial.points }] : []),
    ]

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading
                eyebrow="Análisis por umbral"
                title={curveType === 'roc' ? 'Curvas ROC robustas' : 'Curvas Precision–Recall robustas'}
                description="A y B se comparan directamente; cuando comparten modelo se añade una única referencia original discontinua."
                controls={<div className="evaluation-comparison-segmented"><button className={curveType === 'roc' ? 'active' : ''} onClick={() => setCurveType('roc')} type="button">ROC</button><button className={curveType === 'precision_recall' ? 'active' : ''} onClick={() => setCurveType('precision_recall')} type="button">Precision–Recall</button></div>}
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
            <ComparisonSectionHeading eyebrow="Desglose por categoría" title="AP, AR y supresión por clase" description="Localiza categorías donde una evaluación conserva mejor la calidad o suprime menos objetos." controls={<label><span>Orden</span><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="difference">Mayor diferencia AP</option><option value="support">Mayor soporte</option><option value="name">Nombre</option></select></label>} />
            <div className="evaluation-comparison-search-row"><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar categoría" /><span>{rows.length} clases</span></div>
            <ComparisonDataTable
                columns={[
                    { key: 'label', label: 'Clase', width: '1.45fr', render: (row) => <><strong>{row.label}</strong><small>{row.support} objetos</small></> },
                    { key: 'leftAp', label: 'AP A', className: 'side-a', render: (row) => formatPercent(row.leftAp) },
                    { key: 'rightAp', label: 'AP B', className: 'side-b', render: (row) => formatPercent(row.rightAp) },
                    { key: 'delta', label: 'Δ AP', render: (row) => formatPoints(row.delta) },
                    { key: 'ar', label: 'AR A / B', render: (row) => `${formatPercent(row.leftAr)} / ${formatPercent(row.rightAr)}` },
                    { key: 'suppressed', label: 'Suprimidos A / B', render: (row) => `${row.leftSuppressed} / ${row.rightSuppressed}` },
                ]}
                rows={rows}
            />
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
            <ComparisonCurveChart
                series={[
                    { key: 'a', label: 'Evaluación A', color: 'var(--comparison-a)', points: leftCurve?.points },
                    { key: 'b', label: 'Evaluación B', color: 'var(--comparison-b)', points: rightCurve?.points },
                ]}
                xKey="recall"
                xLabel="Recall"
                yLabel="Precisión"
            />
        </section>
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
            <ComparisonSectionHeading eyebrow="Diagnóstico TIDE" title="Qué tipo de error pesa más en A y B" description="dAP representa el AP50 recuperable si se corrigiera hipotéticamente un único tipo de error." />
            <div className="evaluation-comparison-chart tide" role="img" aria-label="Comparación TIDE A y B">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 10, right: 18, bottom: 18, left: 8 }}>
                        <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="4 4" vertical={false} />
                        <XAxis dataKey="label" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                        <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                        <Tooltip cursor={false} />
                        <Bar dataKey="a" name="Evaluación A" fill="var(--comparison-a)" radius={[5, 5, 0, 0]} />
                        <Bar dataKey="b" name="Evaluación B" fill="var(--comparison-b)" radius={[5, 5, 0, 0]} />
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

function DetectionObjectComparison({ leftEvaluation, rightEvaluation }) {
    const pairs = useMemo(() => getDetectionObjectComparisonPairs(leftEvaluation, rightEvaluation), [leftEvaluation, rightEvaluation])
    const [query, setQuery] = useState('')
    const [page, setPage] = useState(1)
    const pageSize = 25
    const filtered = pairs.filter(({ left }) => !query.trim() || [left.object_id, left.image_id, left.category?.category_name].some((value) => String(value ?? '').toLocaleLowerCase('es').includes(query.trim().toLocaleLowerCase('es'))))
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
    const pageItems = filtered.slice((page - 1) * pageSize, page * pageSize)

    if (!pairs.length) return <EmptyComparison title="Detalle por objeto no disponible">Las dos evaluaciones deben incluir el detalle materializado por objeto.</EmptyComparison>

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Objetos evaluados" title="Transición de cada objeto en A y B" description="Compara el desenlace del mismo objeto anotado y su observación adversaria en ambas evaluaciones." />
            <div className="evaluation-comparison-search-row"><input type="search" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1) }} placeholder="Buscar objeto, imagen o clase" /><span>{filtered.length} objetos</span></div>
            <ComparisonDataTable columns={[
                { key: 'object', label: 'Objeto', width: '1.25fr', render: (row) => <><strong>{row.left.category?.category_name}</strong><small>{row.left.object_id} · {row.left.image_id}</small></> },
                { key: 'a', label: 'A', className: 'side-a', render: (row) => <><strong>{row.left.outcome}</strong><small>{ObjectObservation({ object: row.left })}</small></> },
                { key: 'b', label: 'B', className: 'side-b', render: (row) => <><strong>{row.right.outcome}</strong><small>{ObjectObservation({ object: row.right })}</small></> },
                { key: 'confidence', label: 'Δ confianza A / B', render: (row) => `${formatNumber(row.left.confidence_drop)} / ${formatNumber(row.right.confidence_drop)}` },
                { key: 'iou', label: 'Δ IoU A / B', render: (row) => `${formatNumber(row.left.iou_drop)} / ${formatNumber(row.right.iou_drop)}` },
            ]} rows={pageItems} getKey={(row) => row.left.object_id} />
            {totalPages > 1 && <div className="evaluation-comparison-pagination"><button type="button" disabled={page === 1} onClick={() => setPage((current) => Math.max(1, current - 1))}>Anterior</button><span>Página {page} de {totalPages}</span><button type="button" disabled={page === totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))}>Siguiente</button></div>}
        </section>
    )
}

function ExpandIcon() {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.5 4H4v4.5M15.5 4H20v4.5M20 15.5V20h-4.5M8.5 20H4v-4.5" /></svg>
}

function ComparisonImageLightbox({ preview, onClose }) {
    const [zoom, setZoom] = useState(1)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const viewportRef = useRef(null)
    const imageRef = useRef(null)
    const [dragging, setDragging] = useState(false)
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
        const image = imageRef.current
        if (!viewport || !image) return { x: 0, y: 0 }
        return getEvaluationPreviewPanBounds({ viewportWidth: viewport.clientWidth, viewportHeight: viewport.clientHeight, imageWidth: image.offsetWidth, imageHeight: image.offsetHeight, zoom: nextZoom })
    }
    function changeZoom(direction) {
        const next = stepEvaluationPreviewZoom(zoom, direction)
        setZoom(next)
        setPan((current) => clampEvaluationPreviewPan(current, bounds(next)))
    }

    const modal = (
        <div className="image-lightbox-backdrop evaluation-image-lightbox-backdrop comparison-image-lightbox-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
            <section className="image-lightbox evaluation-image-lightbox comparison-image-lightbox" role="dialog" aria-modal="true">
                <header><div><span className="evaluation-eyebrow">Vista ampliada</span><strong>{preview.label}</strong></div><div className="evaluation-image-lightbox-controls"><button type="button" onClick={() => changeZoom(-1)} disabled={zoom <= 1}>−</button><span>{Math.round(zoom * 100)} %</span><button type="button" onClick={() => changeZoom(1)} disabled={zoom >= 4}>+</button><button type="button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }}>Restablecer</button><button type="button" onClick={onClose}>×</button></div></header>
                <div className={`evaluation-image-lightbox-stage${dragging ? ' is-dragging' : ''}`} ref={viewportRef} onPointerDown={(event) => { if (zoom <= 1) return; event.currentTarget.setPointerCapture(event.pointerId); dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, pan }; setDragging(true) }} onPointerMove={(event) => { const drag = dragRef.current; if (!drag || drag.id !== event.pointerId) return; setPan(clampEvaluationPreviewPan({ x: drag.pan.x + event.clientX - drag.x, y: drag.pan.y + event.clientY - drag.y }, bounds())) }} onPointerUp={() => { dragRef.current = null; setDragging(false) }}>
                    <img ref={imageRef} src={resolveApiUrl(preview.url)} alt={preview.label} draggable="false" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${clampEvaluationPreviewZoom(zoom)})` }} />
                </div>
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
                <img
                    src={resolveApiUrl(url)}
                    alt={label}
                    loading="lazy"
                    onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
                />
                {dimensions && detections?.length > 0 && (
                    <svg className="evaluation-comparison-detection-overlay" viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} aria-hidden="true">
                        {detections.map((detection, index) => {
                            const box = detection.bbox
                            if (!box) return null
                            return <rect key={`${detection.label}-${index}`} x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1} />
                        })}
                    </svg>
                )}
                <button type="button" onClick={() => onExpand({ label, url })} aria-label={`Ampliar ${label}`}><ExpandIcon /></button>
            </div>
        </figure>
    )
}

function ComparisonImages({ taskType, leftEvaluation, rightEvaluation }) {
    const pairs = useMemo(() => getComparisonImagePairs(leftEvaluation, rightEvaluation), [leftEvaluation, rightEvaluation])
    const [visible, setVisible] = useState(8)
    const [preview, setPreview] = useState(null)
    const shown = pairs.slice(0, visible)

    return (
        <section className="evaluation-comparison-detail-section">
            <ComparisonSectionHeading eyebrow="Casos concretos" title="La misma imagen bajo A y B" description="La referencia original se muestra una sola vez; las dos columnas siguientes permiten inspeccionar cómo cambia el resultado adversario." />
            {shown.map(({ left, right }) => {
                const originalUrl = left.original_image_url
                const leftUrl = left.adversarial_image_url
                const rightUrl = right.adversarial_image_url
                return (
                    <article key={left.image_id} className="evaluation-comparison-image-case">
                        <header><div><strong>{left.filename ?? left.image_id}</strong><small>{left.image_id}</small></div><div className="evaluation-comparison-image-outcomes"><span className="side-a">A · {taskType === 'detection' ? `${left.adversarial_detection_count} detecciones` : `${left.adversarial_label} · ${formatPercent(left.adversarial_confidence)}`}</span><span className="side-b">B · {taskType === 'detection' ? `${right.adversarial_detection_count} detecciones` : `${right.adversarial_label} · ${formatPercent(right.adversarial_confidence)}`}</span></div></header>
                        <div className={`evaluation-comparison-image-grid${taskType === 'detection' ? ' detection' : ''}`}>
                            {taskType === 'detection' ? (
                                <>
                                    <ComparisonImageFigure label="A · Original" url={`/datasets/${leftEvaluation.dataset_id}/images/${left.image_id}`} caption={`${left.original_detection_count} cajas`} detections={left.original_detections} onExpand={setPreview} tone="side-a" />
                                    <ComparisonImageFigure label="A · Tras el ataque" url={left.adversarial_image_url} caption={`${left.adversarial_detection_count} cajas`} detections={left.adversarial_detections} onExpand={setPreview} tone="side-a" />
                                    <ComparisonImageFigure label="B · Original" url={`/datasets/${rightEvaluation.dataset_id}/images/${right.image_id}`} caption={`${right.original_detection_count} cajas`} detections={right.original_detections} onExpand={setPreview} tone="side-b" />
                                    <ComparisonImageFigure label="B · Tras el ataque" url={right.adversarial_image_url} caption={`${right.adversarial_detection_count} cajas`} detections={right.adversarial_detections} onExpand={setPreview} tone="side-b" />
                                </>
                            ) : (
                                <>
                                    <ComparisonImageFigure label="Original" url={originalUrl} caption={left.ground_truth_label} onExpand={setPreview} tone="reference" />
                                    <ComparisonImageFigure label="Evaluación A" url={leftUrl} caption={left.adversarial_label} onExpand={setPreview} tone="side-a" />
                                    <ComparisonImageFigure label="Evaluación B" url={rightUrl} caption={right.adversarial_label} onExpand={setPreview} tone="side-b" />
                                </>
                            )}
                        </div>
                    </article>
                )
            })}
            {pairs.length > 8 && <div className="evaluation-comparison-more"><span>{Math.min(visible, pairs.length)} / {pairs.length}</span>{visible < pairs.length ? <button type="button" onClick={() => setVisible((current) => Math.min(pairs.length, current + 8))}>Ver 8 más</button> : <button type="button" onClick={() => setVisible(8)}>Mostrar menos</button>}</div>}
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
                {taskType === 'classification' && activeTab === 'curves' && <ClassificationCurveComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} showReference={sameModel} />}
                {taskType === 'classification' && activeTab === 'images' && <ComparisonImages taskType={taskType} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'coco' && <DetectionCocoComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'classes' && <DetectionClassComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'curves' && <DetectionCurveComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'objects' && <DetectionObjectComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'tide' && <DetectionTideComparison leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
                {taskType === 'detection' && activeTab === 'images' && <ComparisonImages taskType={taskType} leftEvaluation={leftEvaluation} rightEvaluation={rightEvaluation} />}
            </div>
        </div>
    )
}

export default EvaluationComparisonDetails
