import MetricCard from '../evaluation/MetricCard'
import { formatInteger, formatRatio } from '../../utils/metricFormatting'
import {
    COCO_METRIC_ROWS,
    formatSignedPercentagePoints,
    formatSignedPercent,
    getDeltaTone,
    getDetectionUnavailableMessage,
} from '../../utils/detectionMetrics'

function OperatingPoint({ title, metrics, tone }) {
    return (
        <article className={`evaluation-surface detection-operating-detail tone-${tone}`}>
            <h4>{title}</h4>
            <div className="detection-count-strip">
                <div><span>TP</span><strong>{formatInteger(metrics.true_positives)}</strong></div>
                <div><span>FP</span><strong>{formatInteger(metrics.false_positives)}</strong></div>
                <div><span>FN</span><strong>{formatInteger(metrics.false_negatives)}</strong></div>
            </div>
            <dl>
                <div><dt>Precisión</dt><dd>{formatRatio(metrics.precision)}</dd></div>
                <div><dt>Recall</dt><dd>{formatRatio(metrics.recall)}</dd></div>
                <div><dt>F1</dt><dd>{formatRatio(metrics.f1)}</dd></div>
            </dl>
        </article>
    )
}

function DetectionCocoMetrics({ evaluation }) {
    const supervised = evaluation.supervised
    const inferenceMode = supervised?.configuration?.inference_mode
    const usesEndToEndInference = inferenceMode === 'end_to_end_nms_free'

    if (!supervised?.available) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Evaluación COCO</span>
                <h3>Las métricas COCO no están disponibles</h3>
                <p>{getDetectionUnavailableMessage(supervised?.reason)}</p>
            </div>
        )
    }

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-kpi-grid">
                <MetricCard
                    label="AP@0.50 original"
                    value={formatRatio(supervised.original.aggregate.ap_50)}
                    helper="IoU 0.50"
                    tone="original"
                    info="Average Precision con un emparejamiento mínimo IoU 0,50. Resume la curva Precision–Recall a ese umbral."
                />
                <MetricCard
                    label="AP@0.50 tras el ataque"
                    value={formatRatio(supervised.adversarial.aggregate.ap_50)}
                    helper="IoU 0.50"
                    tone="adversarial"
                    info="La misma métrica AP@0.50 calculada sobre las detecciones obtenidas después del ataque."
                />
                <MetricCard
                    label="AR@100 original"
                    value={formatRatio(supervised.original.aggregate.ar_100)}
                    helper="Hasta 100 detecciones"
                    tone="original"
                    info="Average Recall permitiendo hasta 100 detecciones por imagen y promediando los umbrales IoU de COCO."
                />
                <MetricCard
                    label="AR@100 tras el ataque"
                    value={formatRatio(supervised.adversarial.aggregate.ar_100)}
                    helper="Hasta 100 detecciones"
                    tone="adversarial"
                    info="Average Recall tras el ataque bajo la misma configuración COCO de hasta 100 detecciones por imagen."
                />
            </section>

            <section className="detection-table-section detection-coco-comparison">
                <div className="evaluation-section-heading">
                    <div>
                        <span className="evaluation-eyebrow">Comparación global</span>
                        <h3>AP y AR en la evaluación COCO</h3>
                    </div>
                    <div className="evaluation-legend-pills">
                        <span className="original">Original</span>
                        <span className="adversarial">Tras el ataque</span>
                    </div>
                </div>

                <div className="detection-data-grid-scroll">
                    <div className="detection-data-grid detection-coco-grid" role="table" aria-label="Comparación de métricas COCO">
                        <div className="detection-data-grid-header" role="row">
                            <span role="columnheader">Métrica</span>
                            <span role="columnheader" className="tone-original">Original</span>
                            <span role="columnheader" className="tone-adversarial">Tras el ataque</span>
                            <span role="columnheader">Δ absoluto</span>
                            <span role="columnheader">Δ relativo</span>
                        </div>
                        {COCO_METRIC_ROWS.map(([field, label, description]) => {
                            const delta = supervised.deltas.aggregate[field]
                            return (
                                <div className="detection-data-grid-row" role="row" key={field}>
                                    <div className="detection-data-grid-label" role="cell">
                                        <strong>{label}</strong>
                                        <small>{description}</small>
                                    </div>
                                    <span role="cell">{formatRatio(supervised.original.aggregate[field])}</span>
                                    <span role="cell">{formatRatio(supervised.adversarial.aggregate[field])}</span>
                                    <span role="cell" className={`tone-${getDeltaTone(delta.absolute)}`}>
                                        {formatSignedPercentagePoints(delta.absolute)}
                                    </span>
                                    <span role="cell">{formatSignedPercent(delta.relative)}</span>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </section>

            <section className="detection-operating-grid detection-coco-operating-grid">
                <OperatingPoint title="Punto de operación original" metrics={supervised.original.operating_point} tone="original" />
                <OperatingPoint title="Punto de operación tras el ataque" metrics={supervised.adversarial.operating_point} tone="adversarial" />
            </section>

            <section className="detection-method-section">
                <span className="evaluation-eyebrow">Configuración efectiva</span>
                <h3>Cómo se han calculado estas métricas</h3>
                <dl className="detection-method-grid">
                    <div><dt>Confianza mínima evaluada</dt><dd>{supervised.configuration.score_floor}</dd></div>
                    <div><dt>Confianza</dt><dd>≥ {supervised.configuration.confidence_threshold}</dd></div>
                    <div><dt>IoU de emparejamiento</dt><dd>≥ {supervised.configuration.iou_threshold}</dd></div>
                    <div><dt>Inferencia</dt><dd>{usesEndToEndInference ? 'De extremo a extremo (sin NMS)' : 'NMS tradicional'}</dd></div>
                    <div><dt>NMS</dt><dd>{usesEndToEndInference ? 'No se aplica' : `IoU ${supervised.configuration.nms_iou_threshold}`}</dd></div>
                    <div><dt>Máx. detecciones</dt><dd>{supervised.configuration.max_detections_per_image}</dd></div>
                    <div><dt>Recall COCO</dt><dd>{supervised.configuration.coco_recall_threshold_count} puntos</dd></div>
                </dl>
            </section>
        </div>
    )
}

export default DetectionCocoMetrics
