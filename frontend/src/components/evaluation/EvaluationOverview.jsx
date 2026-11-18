import MetricCard from './MetricCard'
import MetricComparisonChart from './MetricComparisonChart'
import { CLASSIFICATION_METRIC_HELP } from '../../utils/evaluationMetricGuides'
import {
    formatAvailableMetric,
    formatDecimal,
    formatInteger,
    formatPoints,
    formatRatio,
} from '../../utils/metricFormatting'

const TRANSITION_COLORS = {
    danger: 'var(--evaluation-adversarial)',
    success: 'var(--evaluation-success)',
    warning: 'var(--evaluation-transition-warning)',
    neutral: '#cbd5e1',
    unresolved: '#64748b',
}

function buildTransitionBreakdown(evaluation, robustness) {
    const results = Array.isArray(evaluation.results) ? evaluation.results : []

    if (!results.length) {
        const otherChanged = Math.max(
            0,
            robustness.changed_predictions
                - robustness.correct_to_incorrect
                - robustness.incorrect_to_correct,
        )

        return [
            {
                key: 'correct_to_incorrect',
                label: 'Correcta → incorrecta',
                count: robustness.correct_to_incorrect,
                tone: 'danger',
            },
            {
                key: 'incorrect_to_correct',
                label: 'Incorrecta → correcta',
                count: robustness.incorrect_to_correct,
                tone: 'success',
            },
            {
                key: 'other_changed',
                label: 'Otros cambios de predicción',
                count: otherChanged,
                tone: 'warning',
            },
            {
                key: 'unchanged',
                label: 'Sin cambio de predicción',
                count: robustness.unchanged_predictions,
                tone: 'neutral',
            },
        ]
    }

    const counts = {
        correctToIncorrect: 0,
        incorrectToCorrect: 0,
        incorrectToIncorrectChanged: 0,
        unresolvedChanged: 0,
        unchanged: 0,
    }

    results.forEach((result) => {
        if (!result.prediction_changed) {
            counts.unchanged += 1
            return
        }

        const transition = result.diagnostics?.transition

        if (transition === 'correct_to_incorrect') {
            counts.correctToIncorrect += 1
        } else if (transition === 'incorrect_to_correct') {
            counts.incorrectToCorrect += 1
        } else if (transition === 'incorrect_to_incorrect') {
            counts.incorrectToIncorrectChanged += 1
        } else {
            counts.unresolvedChanged += 1
        }
    })

    const items = [
        {
            key: 'correct_to_incorrect',
            label: 'Correcta → incorrecta',
            count: counts.correctToIncorrect,
            tone: 'danger',
        },
        {
            key: 'incorrect_to_correct',
            label: 'Incorrecta → correcta',
            count: counts.incorrectToCorrect,
            tone: 'success',
        },
        {
            key: 'incorrect_to_incorrect_changed',
            label: 'Incorrecta → incorrecta, clase distinta',
            count: counts.incorrectToIncorrectChanged,
            tone: 'warning',
        },
        {
            key: 'unchanged',
            label: 'Sin cambio de predicción',
            count: counts.unchanged,
            tone: 'neutral',
        },
    ]

    if (counts.unresolvedChanged > 0) {
        items.splice(3, 0, {
            key: 'unresolved_changed',
            label: 'Cambio sin etiqueta resuelta',
            count: counts.unresolvedChanged,
            tone: 'unresolved',
        })
    }

    return items
}

function buildDonutBackground(items) {
    const total = items.reduce((sum, item) => sum + item.count, 0)

    if (!total) {
        return '#e8edf5'
    }

    let angle = 0
    const segments = items
        .filter((item) => item.count > 0)
        .map((item) => {
            const start = angle
            angle += (item.count / total) * 360
            return `${TRANSITION_COLORS[item.tone]} ${start}deg ${angle}deg`
        })

    return `conic-gradient(${segments.join(', ')})`
}

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
    const transitionItems = buildTransitionBreakdown(evaluation, robustness)
    const donutBackground = buildDonutBackground(transitionItems)

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-kpi-grid" aria-label="Indicadores principales">
                <MetricCard
                    label="Exactitud original"
                    value={formatAvailableMetric(robustness.clean_accuracy)}
                    helper="Rendimiento antes del ataque"
                    tone="original"
                />
                <MetricCard
                    label="Exactitud robusta"
                    value={formatAvailableMetric(robustness.robust_accuracy)}
                    helper="Rendimiento tras el ataque"
                    tone="adversarial"
                />
                <MetricCard
                    label="Caída de exactitud"
                    value={formatAvailableMetric(robustness.accuracy_drop, formatPoints)}
                    helper="Caída en puntos porcentuales"
                    tone="danger"
                />
                <MetricCard
                    label="ASR condicional"
                    value={formatAvailableMetric(
                        robustness.conditional_attack_success_rate,
                    )}
                    helper="Correctas que pasan a incorrectas"
                    tone="danger"
                    info={CLASSIFICATION_METRIC_HELP.conditional_asr}
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
                            <h3>Original frente a tras el ataque</h3>
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
                            style={{ background: donutBackground }}
                            role="img"
                            aria-label={`El ${formatRatio(
                                robustness.prediction_flip_rate,
                                0,
                            )} de las predicciones ha cambiado`}
                        >
                            <div>
                                <strong>{formatRatio(robustness.prediction_flip_rate, 0)}</strong>
                                <span>cambiadas</span>
                            </div>
                        </div>

                        <dl className="evaluation-impact-list" aria-label="Desglose de transiciones">
                            {transitionItems.map((item) => (
                                <div key={item.key} className={`tone-${item.tone}`}>
                                    <dt>{item.label}</dt>
                                    <dd>{formatInteger(item.count)}</dd>
                                </div>
                            ))}
                        </dl>
                    </div>
                </article>
            </section>

            <section className="evaluation-secondary-grid">
                <MetricCard
                    label="F1 macro original"
                    value={aggregate ? formatRatio(aggregate.original.f1_macro) : '—'}
                    helper="Todas las clases pesan lo mismo"
                    tone="original"
                    info={CLASSIFICATION_METRIC_HELP.f1_macro}
                />
                <MetricCard
                    label="F1 macro tras el ataque"
                    value={aggregate ? formatRatio(aggregate.adversarial.f1_macro) : '—'}
                    helper="Tras aplicar la perturbación"
                    tone="adversarial"
                    info={CLASSIFICATION_METRIC_HELP.f1_macro}
                />
                <MetricCard
                    label="Exactitud robusta top-5"
                    value={aggregate ? formatRatio(aggregate.adversarial.top_5_accuracy) : '—'}
                    helper="La clase real permanece entre las cinco primeras"
                    tone="neutral"
                    info={CLASSIFICATION_METRIC_HELP.top5_robust_accuracy}
                />
                <MetricCard
                    label="Caída prob. clase real"
                    value={formatAvailableMetric(
                        robustness.average_true_class_probability_drop,
                    )}
                    helper="Promedio original menos tras el ataque"
                    tone="danger"
                    info={CLASSIFICATION_METRIC_HELP.true_class_probability_drop}
                />
                <MetricCard
                    label="Perturbación media L∞"
                    value={formatDecimal(robustness.average_perturbation_linf)}
                    helper="Máxima variación por píxel"
                    tone="neutral"
                    info={CLASSIFICATION_METRIC_HELP.linf}
                />
                <MetricCard
                    label="Perturbación media L2"
                    value={formatDecimal(robustness.average_perturbation_l2)}
                    helper="Magnitud total de la perturbación"
                    tone="neutral"
                    info={CLASSIFICATION_METRIC_HELP.l2}
                />
            </section>

            <section className="evaluation-surface evaluation-coverage-card">
                <div>
                    <span className="evaluation-eyebrow">Calidad del conjunto de datos</span>
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
