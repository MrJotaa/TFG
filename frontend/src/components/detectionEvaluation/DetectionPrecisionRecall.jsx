import { useMemo, useState } from 'react'
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts'
import { formatRatio } from '../../utils/metricFormatting'
import {
    findClassMetrics,
    getDetectionUnavailableMessage,
    mergePrecisionRecallCurves,
} from '../../utils/detectionMetrics'

const TICKS = [0, 0.25, 0.5, 0.75, 1]

function CurveTooltip({ active, payload }) {
    if (!active || !payload?.length) {
        return null
    }

    const point = payload[0].payload
    return (
        <div className="evaluation-chart-tooltip">
            <strong>Recall {formatRatio(point.recall)}</strong>
            {payload.map((entry) => (
                <div key={entry.dataKey} className={entry.dataKey}>
                    <span>{entry.name}</span>
                    <strong>{formatRatio(entry.value)}</strong>
                </div>
            ))}
        </div>
    )
}

function DetectionPrecisionRecall({ evaluation }) {
    const [scope, setScope] = useState('global')
    const supervised = evaluation.supervised

    const selected = useMemo(() => {
        if (!supervised?.available || scope === 'global') {
            return {
                label: 'Promedio global',
                original: supervised?.original?.precision_recall_curve,
                adversarial: supervised?.adversarial?.precision_recall_curve,
                originalScore: supervised?.original?.aggregate.map_50_95,
                adversarialScore: supervised?.adversarial?.aggregate.map_50_95,
            }
        }

        const original = findClassMetrics(supervised.original, scope)
        const adversarial = findClassMetrics(supervised.adversarial, scope)
        return {
            label: original?.reference.category_name ?? 'Clase',
            original: original?.precision_recall_curve,
            adversarial: adversarial?.precision_recall_curve,
            originalScore: original?.average_precision,
            adversarialScore: adversarial?.average_precision,
        }
    }, [scope, supervised])

    if (!supervised?.available) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Curvas Precision-Recall</span>
                <h3>Las curvas no están disponibles</h3>
                <p>{getDetectionUnavailableMessage(supervised?.reason)}</p>
            </div>
        )
    }

    const data = mergePrecisionRecallCurves(selected.original, selected.adversarial)

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-surface detection-curve-header">
                <div>
                    <span className="evaluation-eyebrow">101 puntos de recall COCO</span>
                    <h3>Curvas Precision-Recall</h3>
                    <p>Compara la envolvente interpolada original y adversarial para el promedio global o una clase.</p>
                </div>
                <label>
                    <span>Serie</span>
                    <select value={scope} onChange={(event) => setScope(event.target.value)}>
                        <option value="global">Promedio global</option>
                        {supervised.original.per_class.map((entry) => (
                            <option key={entry.reference.category_internal_id} value={entry.reference.model_class_id}>
                                {entry.reference.category_name} · n={entry.support}
                            </option>
                        ))}
                    </select>
                </label>
            </section>

            <section className="detection-curve-score-grid">
                <article className="curve-score-card original"><span>AP original</span><strong>{formatRatio(selected.originalScore)}</strong><small>{selected.label}</small></article>
                <article className="curve-score-card adversarial"><span>AP adversarial</span><strong>{formatRatio(selected.adversarialScore)}</strong><small>{selected.label}</small></article>
            </section>

            {data.length ? (
                <section className="evaluation-surface detection-pr-chart" role="img" aria-label={`Curva Precision-Recall para ${selected.label}`}>
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data} margin={{ top: 16, right: 24, bottom: 22, left: 16 }}>
                            <CartesianGrid stroke="#e7edf5" strokeDasharray="4 4" />
                            <XAxis type="number" dataKey="recall" domain={[0, 1]} ticks={TICKS} tickFormatter={(value) => Number(value).toFixed(2)} label={{ value: 'Recall', position: 'insideBottom', offset: -12 }} />
                            <YAxis type="number" domain={[0, 1]} ticks={TICKS} tickFormatter={(value) => Number(value).toFixed(2)} label={{ value: 'Precisión', angle: -90, position: 'insideLeft' }} />
                            <Tooltip content={<CurveTooltip />} />
                            <Line dataKey="original" name="Original" stroke="var(--evaluation-original)" strokeWidth={3} dot={false} isAnimationActive={false} />
                            <Line dataKey="adversarial" name="Adversarial" stroke="var(--evaluation-adversarial)" strokeWidth={3} dot={false} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </section>
            ) : (
                <div className="evaluation-empty-panel compact">
                    <h3>Curva no definida</h3>
                    <p>Esta clase no tiene soporte evaluable para construir una curva.</p>
                </div>
            )}
        </div>
    )
}

export default DetectionPrecisionRecall
