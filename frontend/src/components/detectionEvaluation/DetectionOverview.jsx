import MetricCard from '../evaluation/MetricCard'
import {
    formatDecimal,
    formatInteger,
    formatRatio,
} from '../../utils/metricFormatting'
import {
    formatSignedPercentagePoints,
    getDetectionUnavailableMessage,
    getObjectOutcomeItems,
} from '../../utils/detectionMetrics'

function DetectionOverview({ evaluation }) {
    const { summary, supervised } = evaluation
    const objectSummary = supervised?.object_analysis?.summary
    const outcomes = getObjectOutcomeItems(objectSummary)
    const outcomeTotal = objectSummary?.total_objects ?? 0

    return (
        <div className="evaluation-tab-content">
            {supervised?.available ? (
                <>
                    <section className="evaluation-kpi-grid" aria-label="Indicadores COCO principales">
                        <MetricCard
                            label="mAP original"
                            value={formatRatio(supervised.original.aggregate.map_50_95)}
                            helper="COCO mAP entre IoU 0.50 y 0.95"
                            tone="original"
                        />
                        <MetricCard
                            label="mAP tras el ataque"
                            value={formatRatio(supervised.adversarial.aggregate.map_50_95)}
                            helper="Rendimiento después del ataque"
                            tone="adversarial"
                        />
                        <MetricCard
                            label="Degradación de mAP"
                            value={formatSignedPercentagePoints(
                                supervised.deltas.aggregate.map_50_95.absolute,
                            )}
                            helper="Original menos resultado tras el ataque"
                            tone="danger"
                        />
                        <MetricCard
                            label="Supresión de objetos"
                            value={formatRatio(objectSummary?.suppression_rate)}
                            helper="Detectados originalmente que desaparecen"
                            tone="danger"
                        />
                    </section>

                    {supervised.warning && (
                        <div className="evaluation-warning-banner">
                            <strong>Cobertura parcial.</strong>
                            <span>{supervised.warning}</span>
                        </div>
                    )}

                    <section className="evaluation-overview-grid">
                        <article className="evaluation-surface detection-outcomes-card">
                            <div className="evaluation-section-heading">
                                <div>
                                    <span className="evaluation-eyebrow">Transiciones por objeto</span>
                                    <h3>Objetos de referencia antes y después del ataque</h3>
                                </div>
                            </div>

                            <div className="detection-outcome-bar" aria-label="Distribución de outcomes">
                                {outcomes.map((item) => (
                                    <span
                                        key={item.field}
                                        className={`tone-${item.tone}`}
                                        style={{
                                            width: outcomeTotal
                                                ? `${(item.count / outcomeTotal) * 100}%`
                                                : '0%',
                                        }}
                                    />
                                ))}
                            </div>

                            <dl className="detection-outcome-list">
                                {outcomes.map((item) => (
                                    <div key={item.field} className={`tone-${item.tone}`}>
                                        <dt>{item.label}</dt>
                                        <dd>{formatInteger(item.count)}</dd>
                                    </div>
                                ))}
                            </dl>
                        </article>

                        <article className="evaluation-surface detection-operating-card">
                            <div className="evaluation-section-heading">
                                <div>
                                    <span className="evaluation-eyebrow">Punto de operación</span>
                                    <h3>Calidad a umbral fijo</h3>
                                </div>
                            </div>

                            <div className="detection-operating-comparison">
                                <div>
                                    <span>Original</span>
                                    <strong>{formatRatio(supervised.original.operating_point.f1)}</strong>
                                    <small>F1</small>
                                </div>
                                <div>
                                    <span>Tras el ataque</span>
                                    <strong>{formatRatio(supervised.adversarial.operating_point.f1)}</strong>
                                    <small>F1</small>
                                </div>
                            </div>

                            <p>
                                Confianza ≥ {formatDecimal(supervised.configuration.confidence_threshold, 2)} · IoU de emparejamiento ≥{' '}
                                {formatDecimal(supervised.configuration.iou_threshold, 2)}
                            </p>
                        </article>
                    </section>
                </>
            ) : (
                <div className="evaluation-warning-banner">
                    <strong>Métricas supervisadas no disponibles.</strong>
                    <span>{getDetectionUnavailableMessage(supervised?.reason)}</span>
                </div>
            )}

            <section className="evaluation-secondary-grid" aria-label="Métricas descriptivas históricas">
                <MetricCard
                    label="Detecciones originales"
                    value={formatInteger(summary.total_original_detections)}
                    helper="Predicciones publicadas con confianza > 0.25"
                    tone="original"
                />
                <MetricCard
                    label="Detecciones tras el ataque"
                    value={formatInteger(summary.total_adversarial_detections)}
                    helper="No equivale por sí sola a pérdida de calidad"
                    tone="adversarial"
                />
                <MetricCard
                    label="Indicador heurístico de éxito"
                    value={formatRatio(summary.attack_success_rate)}
                    helper="Reducción de conteo o confianza media"
                    tone="neutral"
                />
                <MetricCard
                    label="Confianza media original"
                    value={formatDecimal(summary.average_original_confidence)}
                    helper="Sobre detecciones publicadas"
                    tone="neutral"
                />
                <MetricCard
                    label="Confianza media tras el ataque"
                    value={formatDecimal(summary.average_adversarial_confidence)}
                    helper="Sobre detecciones publicadas"
                    tone="neutral"
                />
                <MetricCard
                    label="Perturbación media L∞"
                    value={formatDecimal(summary.average_perturbation_linf)}
                    helper="Máxima variación por píxel"
                    tone="neutral"
                />
            </section>

            {supervised?.annotation_coverage && (
                <section className="evaluation-surface detection-coverage-grid">
                    <div>
                        <span className="evaluation-eyebrow">Cobertura científica</span>
                        <h3>Anotaciones evaluables</h3>
                        <p>
                            {formatInteger(supervised.annotation_coverage.reviewed_images)} de{' '}
                            {formatInteger(supervised.annotation_coverage.total_images)} imágenes revisadas;{' '}
                            {formatInteger(supervised.annotation_coverage.evaluable_objects)} de{' '}
                            {formatInteger(supervised.annotation_coverage.total_objects)} objetos son evaluables.
                        </p>
                    </div>
                    <div className="detection-coverage-meters">
                        <label>
                            <span>Cobertura de imágenes</span>
                            <progress
                                max="100"
                                value={supervised.annotation_coverage.coverage_percentage}
                            />
                            <strong>{supervised.annotation_coverage.coverage_percentage.toFixed(1)} %</strong>
                        </label>
                        <label>
                            <span>Cobertura de objetos</span>
                            <progress
                                max="100"
                                value={supervised.annotation_coverage.object_coverage_percentage}
                            />
                            <strong>{supervised.annotation_coverage.object_coverage_percentage.toFixed(1)} %</strong>
                        </label>
                    </div>
                </section>
            )}
        </div>
    )
}

export default DetectionOverview
