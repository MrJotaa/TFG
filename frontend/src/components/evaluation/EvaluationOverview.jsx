import MetricCard from './MetricCard'
import MetricComparisonChart from './MetricComparisonChart'
import {
    formatAvailableMetric,
    formatDecimal,
    formatInteger,
    formatPoints,
    formatRatio,
} from '../../utils/metricFormatting'

function EvaluationOverview({ evaluation }) {
    const advancedMetrics = evaluation.advanced_metrics

    if (!advancedMetrics) {
        return (
            <div className="evaluation-empty-panel">
                <h3>Métricas avanzadas no disponibles</h3>
                <p>
                    La respuesta de la evaluación no incluye el contrato avanzado de métricas.
                </p>
            </div>
        )
    }

    const { robustness, label_resolution: labelResolution, supervised_status: status } =
        advancedMetrics
    const aggregate = advancedMetrics.supervised?.aggregate

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-kpi-grid" aria-label="Indicadores principales">
                <MetricCard
                    label="Clean Accuracy"
                    value={formatAvailableMetric(robustness.clean_accuracy)}
                    helper="Rendimiento antes del ataque"
                    tone="original"
                />
                <MetricCard
                    label="Robust Accuracy"
                    value={formatAvailableMetric(robustness.robust_accuracy)}
                    helper="Rendimiento tras el ataque"
                    tone="adversarial"
                />
                <MetricCard
                    label="Accuracy Drop"
                    value={formatAvailableMetric(robustness.accuracy_drop, formatPoints)}
                    helper="Caída en puntos porcentuales"
                    tone="danger"
                />
                <MetricCard
                    label="Conditional ASR"
                    value={formatAvailableMetric(
                        robustness.conditional_attack_success_rate,
                    )}
                    helper="Correctas que pasan a incorrectas"
                    tone="danger"
                />
            </section>

            {!status.available && (
                <div className="evaluation-warning-banner">
                    <strong>Métricas supervisadas no disponibles.</strong>
                    <span>{status.reason}</span>
                </div>
            )}

            {status.warning && (
                <div className="evaluation-warning-banner">
                    <strong>Cobertura parcial de etiquetas.</strong>
                    <span>{status.warning}</span>
                </div>
            )}

            <section className="evaluation-overview-grid">
                <article className="evaluation-surface evaluation-comparison-card">
                    <div className="evaluation-section-heading">
                        <div>
                            <span className="evaluation-eyebrow">Comparación global</span>
                            <h3>Original frente a adversarial</h3>
                        </div>
                        <div className="evaluation-legend-pills" aria-label="Leyenda">
                            <span className="original">Original</span>
                            <span className="adversarial">Adversarial</span>
                        </div>
                    </div>

                    {aggregate ? (
                        <MetricComparisonChart aggregate={aggregate} />
                    ) : (
                        <p className="muted-text">No hay métricas agregadas disponibles.</p>
                    )}
                </article>

                <article className="evaluation-surface evaluation-impact-card">
                    <div className="evaluation-section-heading">
                        <div>
                            <span className="evaluation-eyebrow">Impacto del ataque</span>
                            <h3>Transiciones de predicción</h3>
                        </div>
                    </div>

                    <div className="evaluation-impact-visual">
                        <div
                            className="evaluation-donut"
                            style={{
                                '--changed-angle': `${robustness.prediction_flip_rate * 360}deg`,
                            }}
                        >
                            <div>
                                <strong>{formatRatio(robustness.prediction_flip_rate, 0)}</strong>
                                <span>cambiadas</span>
                            </div>
                        </div>

                        <dl className="evaluation-impact-list">
                            <div>
                                <dt>Correcta → incorrecta</dt>
                                <dd>{formatInteger(robustness.correct_to_incorrect)}</dd>
                            </div>
                            <div>
                                <dt>Incorrecta → correcta</dt>
                                <dd>{formatInteger(robustness.incorrect_to_correct)}</dd>
                            </div>
                            <div>
                                <dt>Sin cambio de clase</dt>
                                <dd>{formatInteger(robustness.unchanged_predictions)}</dd>
                            </div>
                        </dl>
                    </div>
                </article>
            </section>

            <section className="evaluation-secondary-grid">
                <MetricCard
                    label="F1 macro original"
                    value={aggregate ? formatRatio(aggregate.original.f1_macro) : '—'}
                    helper="Todas las clases pesan lo mismo"
                    tone="neutral"
                />
                <MetricCard
                    label="F1 macro adversarial"
                    value={aggregate ? formatRatio(aggregate.adversarial.f1_macro) : '—'}
                    helper="Tras aplicar la perturbación"
                    tone="neutral"
                />
                <MetricCard
                    label="Top-5 robust accuracy"
                    value={aggregate ? formatRatio(aggregate.adversarial.top_5_accuracy) : '—'}
                    helper="La clase real permanece entre las cinco primeras"
                    tone="neutral"
                />
                <MetricCard
                    label="Caída prob. clase real"
                    value={formatAvailableMetric(
                        robustness.average_true_class_probability_drop,
                    )}
                    helper="Promedio original menos adversarial"
                    tone="neutral"
                />
                <MetricCard
                    label="Perturbación media L∞"
                    value={formatDecimal(robustness.average_perturbation_linf)}
                    helper="Máxima variación por píxel"
                    tone="neutral"
                />
                <MetricCard
                    label="Perturbación media L2"
                    value={formatDecimal(robustness.average_perturbation_l2)}
                    helper="Magnitud total de la perturbación"
                    tone="neutral"
                />
            </section>

            <section className="evaluation-surface evaluation-coverage-card">
                <div>
                    <span className="evaluation-eyebrow">Calidad del dataset</span>
                    <h3>Cobertura de etiquetas</h3>
                    <p>
                        {formatInteger(labelResolution.resolved_labels)} de{' '}
                        {formatInteger(labelResolution.labeled_images)} etiquetas se han resuelto
                        contra el catálogo del modelo activo.
                    </p>
                </div>

                <div className="evaluation-coverage-meter" aria-label="Cobertura de etiquetas">
                    <div>
                        <span style={{ width: `${labelResolution.coverage * 100}%` }} />
                    </div>
                    <strong>{formatRatio(labelResolution.coverage)}</strong>
                </div>
            </section>
        </div>
    )
}

export default EvaluationOverview
