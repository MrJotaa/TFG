import {
    Bar,
    BarChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts'
import MetricCard from '../evaluation/MetricCard'
import { formatInteger } from '../../utils/metricFormatting'
import {
    formatTidePoints,
    getDeltaTone,
    getDetectionUnavailableMessage,
    getTideErrorRows,
    TIDE_ERROR_DEFINITIONS,
} from '../../utils/detectionMetrics'

function TideTooltip({ active, payload }) {
    if (!active || !payload?.length) {
        return null
    }
    const row = payload[0].payload
    return (
        <div className="evaluation-chart-tooltip tide-tooltip">
            <strong>{row.label}</strong>
            <p>{row.description}</p>
            {payload.map((entry) => (
                <div key={entry.dataKey} className={entry.dataKey}>
                    <span>{entry.name}</span>
                    <strong>{formatTidePoints(entry.value)}</strong>
                </div>
            ))}
        </div>
    )
}

function DetectionTideAnalysis({ evaluation }) {
    const supervised = evaluation.supervised
    const tide = supervised?.tide_analysis

    if (!supervised?.available || !tide) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Diagnóstico TIDE</span>
                <h3>El análisis TIDE no está disponible</h3>
                <p>{getDetectionUnavailableMessage(supervised?.reason)}</p>
            </div>
        )
    }

    const rows = getTideErrorRows(tide)
    const dominant = TIDE_ERROR_DEFINITIONS[tide.dominant_adversarial_error]
    const specialRows = tide.original.special_errors.map((original) => ({
        definition: TIDE_ERROR_DEFINITIONS[original.error_type],
        original,
        adversarial: tide.adversarial.special_errors.find(
            (item) => item.error_type === original.error_type,
        ),
    }))
    const chartData = rows.map((row) => ({
        ...row,
        originalImpact: row.original.delta_ap,
        adversarialImpact: row.adversarial.delta_ap,
    }))

    return (
        <div className="evaluation-tab-content tide-analysis">
            <section className="evaluation-kpi-grid">
                <MetricCard
                    label="AP50 TIDE original"
                    value={`${tide.original.ap_50.toFixed(1)} %`}
                    helper="AP calculado por TIDE con IoU 0,50"
                    tone="original"
                />
                <MetricCard
                    label="AP50 TIDE tras el ataque"
                    value={`${tide.adversarial.ap_50.toFixed(1)} %`}
                    helper="Misma configuración después del ataque"
                    tone="adversarial"
                />
                <MetricCard
                    label="Error dominante"
                    value={dominant?.label ?? 'Sin impacto aislado'}
                    helper={dominant
                        ? 'Mayor dAP tras el ataque; no necesariamente el más frecuente'
                        : 'Ningún oráculo individual recupera AP50 en esta evaluación'}
                    tone="danger"
                />
                <MetricCard
                    label="Umbrales TIDE"
                    value={`IoU ${tide.configuration.iou_threshold.toFixed(2)}`}
                    helper={`Fondo < ${tide.configuration.background_threshold.toFixed(2)} · máx. ${tide.configuration.max_detections_per_image}`}
                />
            </section>

            <section className="evaluation-surface tide-chart-card">
                <div className="evaluation-section-heading">
                    <div>
                        <span className="evaluation-eyebrow">Impacto aislado</span>
                        <h3>dAP por tipo de error</h3>
                        <p>Altura = AP50 que se recuperaría si solo se corrigiera ese error.</p>
                    </div>
                    <div className="evaluation-legend-pills">
                        <span className="original">Original</span>
                        <span className="adversarial">Tras el ataque</span>
                    </div>
                </div>
                <div className="tide-chart" role="img" aria-label="Comparación del impacto dAP de los errores TIDE">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 12, right: 20, bottom: 18, left: 8 }}>
                            <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
                            <XAxis dataKey="shortLabel" interval={0} tick={{ fontSize: 12 }} />
                            <YAxis tickFormatter={(value) => `${value}`} label={{ value: 'dAP (pp)', angle: -90, position: 'insideLeft' }} />
                            <Tooltip cursor={false} content={<TideTooltip />} />
                            <Bar dataKey="originalImpact" name="Original" fill="var(--evaluation-original)" radius={[5, 5, 0, 0]} />
                            <Bar dataKey="adversarialImpact" name="Tras el ataque" fill="var(--evaluation-adversarial)" radius={[5, 5, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </section>

            <section className="detection-table-section detection-tide-comparison">
                <div className="evaluation-section-heading">
                    <div>
                        <span className="evaluation-eyebrow">Impacto y frecuencia</span>
                        <h3>Cómo cambia la naturaleza de los errores</h3>
                        <p>El conteo indica frecuencia; dAP indica cuánto afecta ese error al AP50.</p>
                    </div>
                </div>
                <div className="detection-data-grid-scroll">
                    <div className="detection-data-grid detection-tide-grid" role="table" aria-label="Impacto y frecuencia de errores TIDE">
                        <div className="detection-data-grid-header" role="row">
                            <span role="columnheader">Error</span>
                            <span role="columnheader" className="tone-original">dAP orig.</span>
                            <span role="columnheader" className="tone-adversarial">dAP adv.</span>
                            <span role="columnheader">Cambio de impacto</span>
                            <span role="columnheader" className="tone-original">Casos orig.</span>
                            <span role="columnheader" className="tone-adversarial">Casos adv.</span>
                            <span role="columnheader">Cambio de casos</span>
                        </div>
                        {rows.map((row) => (
                            <div className="detection-data-grid-row" role="row" key={row.errorType}>
                                <div className="detection-data-grid-label" role="cell">
                                    <strong>{row.label}</strong>
                                    <small>{row.description}</small>
                                </div>
                                <span role="cell">{formatTidePoints(row.original.delta_ap)}</span>
                                <span role="cell">{formatTidePoints(row.adversarial.delta_ap)}</span>
                                <span role="cell" className={`tone-${getDeltaTone(row.change.impact_change)}`}>
                                    {formatTidePoints(row.change.impact_change, 1, true)}
                                </span>
                                <span role="cell">{formatInteger(row.original.count)}</span>
                                <span role="cell">{formatInteger(row.adversarial.count)}</span>
                                <span role="cell">{row.change.count_change > 0 ? '+' : ''}{formatInteger(row.change.count_change)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <section className="tide-special-grid">
                {specialRows.map(({ definition, original, adversarial }) => (
                    <article key={original.error_type} className="evaluation-surface tide-special-card">
                        <span className="evaluation-eyebrow">Oráculo global</span>
                        <h3>{definition.label}</h3>
                        <p>{definition.description}</p>
                        <dl>
                            <div><dt>Original</dt><dd>{formatTidePoints(original.delta_ap)} · {formatInteger(original.count)} casos</dd></div>
                            <div><dt>Tras el ataque</dt><dd>{formatTidePoints(adversarial.delta_ap)} · {formatInteger(adversarial.count)} casos</dd></div>
                        </dl>
                    </article>
                ))}
            </section>

            <section className="evaluation-surface detection-method-card">
                <span className="evaluation-eyebrow">Interpretación correcta</span>
                <h3>TIDE complementa, no sustituye, el mAP COCO</h3>
                <p>Los oráculos corrigen hipotéticamente un tipo de fallo y vuelven a calcular AP50. Sus dAP no deben sumarse: los efectos no son independientes entre sí. Un cambio negativo tampoco implica por sí solo una mejora; el ataque puede haber transformado ese error en otro más grave.</p>
            </section>
        </div>
    )
}

export default DetectionTideAnalysis
