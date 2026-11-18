import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
    getAllEvaluationHistory,
    getStoredEvaluation,
} from '../api/evaluationHistory'
import { useExperiment } from '../context/ExperimentContext'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import {
    baselineConsistency,
    classifyEvaluationComparison,
    getComparisonRetention,
    getComparisonInsights,
    getClassificationClassComparison,
    getDetectionClassComparison,
    pairComparisonMetrics,
} from '../utils/evaluationComparison'
import { formatCreatedAgo } from '../utils/evaluationHistory'
import EvaluationComparisonDetails from '../components/evaluationComparison/EvaluationComparisonDetails'

function formatMetric(value, format) {
    if (value == null || !Number.isFinite(value)) return '—'
    if (format === 'points') return `${(value * 100).toFixed(1)} pp`
    return `${(value * 100).toFixed(1)} %`
}

function formatDelta(value) {
    if (value == null || !Number.isFinite(value)) return '—'
    const scaled = value * 100
    const prefix = scaled > 0 ? '+' : ''
    return `${prefix}${scaled.toFixed(1)} pp`
}

function metricBarWidth(value) {
    if (value == null || !Number.isFinite(value)) return '0%'
    return `${Math.max(0, Math.min(1, value)) * 100}%`
}

function formatSupervisedProtocol(configuration) {
    if (!configuration) return null
    const confidence = configuration.confidence_threshold
    const iou = configuration.iou_threshold
    if (!Number.isFinite(confidence) || !Number.isFinite(iou)) return null
    return `Confianza ≥ ${Number(confidence).toFixed(2)} · IoU ≥ ${Number(iou).toFixed(2)}`
}


function comparisonGuideSummary(comparisonKind) {
    if (comparisonKind === 'model') return 'Compara el rendimiento robusto y su retención frente a la referencia de cada modelo. Un valor absoluto mayor y una menor pérdida responden a preguntas complementarias.'
    if (comparisonKind === 'attack') return 'Compara cuánto rendimiento conserva el mismo modelo frente a cada ataque. Las diferencias describen el impacto observado, pero no convierten automáticamente un ataque en «mejor» o «peor» porque sus intensidades pueden no ser equivalentes.'
    return 'Compara cómo cambia el impacto del mismo ataque al modificar su configuración. La referencia original sirve para interpretar la degradación.'
}

function SummaryComparisonGuide({ taskType, comparisonKind }) {
    const summary = comparisonGuideSummary(comparisonKind)
    return (
        <details className="evaluation-comparison-guide summary-guide">
            <summary>
                <span className="evaluation-comparison-guide-icon" aria-hidden="true">i</span>
                <span><strong>Guía de esta comparación</strong><small>Cómo leer el resumen A/B</small></span>
                <span className="evaluation-comparison-guide-chevron" aria-hidden="true">⌄</span>
            </summary>
            <div className="evaluation-comparison-guide-body">
                <p>{summary}</p>
                <div className="evaluation-comparison-guide-grid">
                    <div><strong>Identificación de las evaluaciones</strong><span>Ámbar y cian solo distinguen evaluaciones; no significan mejor o peor.</span></div>
                    <div><strong>{taskType === 'detection' ? 'Retención de mAP' : 'Retención de exactitud'}</strong><span>Relaciona el resultado robusto con el rendimiento original de la misma evaluación y evita confundir un buen punto de partida con mayor resistencia.</span></div>
                </div>
                <p className="evaluation-comparison-guide-note"><strong>Clave de interpretación:</strong> no se declara un ganador global. Cada métrica responde a una pregunta distinta.</p>
            </div>
        </details>
    )
}

function RetentionProfile({ side, taskType, data }) {
    const label = taskType === 'detection' ? 'mAP' : 'Exactitud'
    const width = (value) => value == null ? '0%' : `${Math.max(0, Math.min(1, value)) * 100}%`
    return (
        <article className={`evaluation-comparison-retention-card side-${side.toLowerCase()}`}>
            <header><span>{side}</span><strong>{label}</strong><small>{data.retention == null ? 'Retención no disponible' : `${(data.retention * 100).toFixed(1)} % retenido`}</small></header>
            <div className="evaluation-comparison-retention-bars">
                <div><span>Original</span><i><b style={{ width: width(data.original) }} /></i><strong>{formatMetric(data.original, 'ratio')}</strong></div>
                <div><span>Robusto</span><i><b style={{ width: width(data.robust) }} /></i><strong>{formatMetric(data.robust, 'ratio')}</strong></div>
            </div>
        </article>
    )
}

function EvaluationIdentity({ label, item, attackCatalog }) {
    return (
        <article
            className={`evaluation-comparison-identity side-${label.toLowerCase()}`}
        >
            <div className="evaluation-comparison-identity-heading">
                <span>{label}</span>
                <small>{formatCreatedAgo(item.created_at)}</small>
            </div>
            <strong>{item.model_name}</strong>
            <div className="evaluation-comparison-attack">
                <span>{getAttackName(item.attack_type, attackCatalog)}</span>
                <small>
                    {formatAttackParameters(
                        item.attack_type,
                        item.parameters,
                        attackCatalog,
                    )}
                </small>
            </div>
        </article>
    )
}

function EvaluationComparisonView() {
    const navigate = useNavigate()
    const [searchParams] = useSearchParams()
    const { attackCatalog } = useExperiment()
    const [state, setState] = useState({ status: 'loading' })

    const leftId = searchParams.get('a')
    const rightId = searchParams.get('b')

    useEffect(() => {
        let cancelled = false

        async function loadComparison() {
            if (!leftId || !rightId || leftId === rightId) {
                setState({
                    status: 'error',
                    message: 'Selecciona dos evaluaciones diferentes desde el historial.',
                })
                return
            }

            try {
                setState({ status: 'loading' })
                const history = await getAllEvaluationHistory()
                const leftItem = history.find((item) => item.result_id === leftId)
                const rightItem = history.find((item) => item.result_id === rightId)

                if (!leftItem || !rightItem) {
                    throw new Error(
                        'Una de las evaluaciones ya no está disponible en el historial.',
                    )
                }

                const comparison = classifyEvaluationComparison(leftItem, rightItem)
                if (!comparison.compatible) {
                    throw new Error(comparison.reason)
                }

                const [leftStored, rightStored] = await Promise.all([
                    getStoredEvaluation(leftId),
                    getStoredEvaluation(rightId),
                ])

                if (
                    leftStored.task_type !== leftItem.task_type
                    || rightStored.task_type !== rightItem.task_type
                ) {
                    throw new Error(
                        'El tipo de una evaluación almacenada no coincide con su historial.',
                    )
                }

                if (cancelled) return
                setState({
                    status: 'ready',
                    leftItem,
                    rightItem,
                    comparison,
                    leftEvaluation: leftStored.evaluation,
                    rightEvaluation: rightStored.evaluation,
                })
            } catch (error) {
                if (!cancelled) {
                    setState({
                        status: 'error',
                        message: error.message || 'No se pudo preparar la comparación.',
                    })
                }
            }
        }

        loadComparison()
        return () => {
            cancelled = true
        }
    }, [leftId, rightId])

    const metrics = useMemo(() => {
        if (state.status !== 'ready') return []
        return pairComparisonMetrics(
            state.leftItem.task_type,
            state.leftEvaluation,
            state.rightEvaluation,
        )
    }, [state])

    if (state.status === 'loading') {
        return (
            <section className="evaluation-comparison-page">
                <div className="evaluation-comparison-loading">
                    <strong>Preparando comparación…</strong>
                    <span>Recuperando las dos evaluaciones sin volver a ejecutarlas.</span>
                </div>
            </section>
        )
    }

    if (state.status === 'error') {
        return (
            <section className="evaluation-comparison-page">
                <div className="evaluation-comparison-error">
                    <span className="evaluation-eyebrow">Comparación de evaluaciones</span>
                    <h1>No se puede construir esta comparación</h1>
                    <p>{state.message}</p>
                    <button
                        type="button"
                        className="primary-action"
                        onClick={() => navigate('/dashboard', { state: { workspaceTab: 'history' } })}
                    >
                        Volver al historial
                    </button>
                </div>
            </section>
        )
    }

    const {
        leftItem,
        rightItem,
        leftEvaluation,
        rightEvaluation,
        comparison,
    } = state
    const taskLabel = leftItem.task_type === 'detection' ? 'detección' : 'clasificación'
    const sameModel = leftItem.model_id === rightItem.model_id
    const baseline = sameModel
        ? baselineConsistency(
            leftItem.task_type,
            leftEvaluation,
            rightEvaluation,
        )
        : null
    const supervisedProtocol = leftItem.task_type === 'detection'
        ? formatSupervisedProtocol(leftItem.supervised_configuration)
        : null
    const leftRetention = getComparisonRetention(leftItem.task_type, leftEvaluation)
    const rightRetention = getComparisonRetention(rightItem.task_type, rightEvaluation)
    const comparisonClassRows = leftItem.task_type === 'detection'
        ? getDetectionClassComparison(leftEvaluation, rightEvaluation)
        : getClassificationClassComparison(leftEvaluation, rightEvaluation, 'f1')
    const insights = getComparisonInsights(leftItem.task_type, metrics, comparisonClassRows)

    return (
        <section className="evaluation-comparison-page">
            <header className="evaluation-comparison-hero">
                <div>
                    <span className="evaluation-eyebrow">
                        Comparación de evaluaciones · {taskLabel}
                    </span>
                    <h1>Comparación de {comparison.label.toLowerCase()}</h1>
                    <p>{comparison.description}</p>
                </div>
                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard', { state: { workspaceTab: 'history' } })}
                >
                    Cambiar evaluaciones
                </button>
            </header>

            <section className="evaluation-comparison-scope">
                <div>
                    <span>Variable analizada</span>
                    <strong>{comparison.label}</strong>
                </div>
                <div>
                    <span>Conjunto de datos constante</span>
                    <strong>{leftItem.dataset_name}</strong>
                    <small>Revisión {leftItem.dataset_revision}</small>
                </div>
                {leftItem.model_id === rightItem.model_id && (
                    <div>
                        <span>Modelo constante</span>
                        <strong>{leftItem.model_name}</strong>
                    </div>
                )}
                {leftItem.attack_type === rightItem.attack_type && (
                    <div>
                        <span>Ataque constante</span>
                        <strong>{getAttackName(leftItem.attack_type, attackCatalog)}</strong>
                        {comparison.kind === 'model' && (
                            <small>
                                {formatAttackParameters(
                                    leftItem.attack_type,
                                    leftItem.parameters,
                                    attackCatalog,
                                )}
                            </small>
                        )}
                    </div>
                )}
                {supervisedProtocol && (
                    <div>
                        <span>Protocolo de evaluación</span>
                        <strong>{supervisedProtocol}</strong>
                    </div>
                )}
            </section>

            <section className="evaluation-comparison-identities" aria-label="Evaluaciones comparadas">
                <EvaluationIdentity label="A" item={leftItem} attackCatalog={attackCatalog} />
                <div className="evaluation-comparison-versus" aria-hidden="true">VS</div>
                <EvaluationIdentity label="B" item={rightItem} attackCatalog={attackCatalog} />
            </section>

            {baseline?.available && (
                <div
                    className={`evaluation-comparison-baseline ${baseline.consistent ? 'consistent' : 'warning'}`}
                >
                    <span aria-hidden="true" />
                    <div>
                        <strong>
                            {baseline.consistent
                                ? 'Referencia original consistente'
                                : 'La referencia original no coincide'}
                        </strong>
                        <small>
                            {baseline.consistent
                                ? 'Ambas evaluaciones parten del mismo rendimiento original.'
                                : 'Revisa la configuración antes de atribuir la diferencia únicamente a la variable analizada.'}
                        </small>
                    </div>
                </div>
            )}

            <EvaluationComparisonDetails
                taskType={leftItem.task_type}
                leftItem={leftItem}
                rightItem={rightItem}
                leftEvaluation={leftEvaluation}
                rightEvaluation={rightEvaluation}
                summary={(
                    <section className="evaluation-comparison-summary">
                        <header>
                            <span className="evaluation-eyebrow">Resumen comparativo</span>
                            <h2>Qué cambia entre A y B</h2>
                            <p>La visualización permite percibir primero el patrón; los valores exactos y el delta B − A permanecen visibles para cuantificarlo.</p>
                        </header>
                        <SummaryComparisonGuide taskType={leftItem.task_type} comparisonKind={comparison.kind} />
                        <section className="evaluation-comparison-retention" aria-label="Retención de rendimiento">
                            <div><span className="evaluation-eyebrow">Perfil de robustez</span><h3>{leftItem.task_type === 'detection' ? 'Conservación de mAP' : 'Conservación de exactitud'}</h3><p>Compara el nivel original y el robusto dentro de cada evaluación antes de mirar el resto de métricas.</p></div>
                            <div className="evaluation-comparison-retention-grid"><RetentionProfile side="A" taskType={leftItem.task_type} data={leftRetention} /><RetentionProfile side="B" taskType={leftItem.task_type} data={rightRetention} /></div>
                        </section>

                        {insights.length > 0 && (
                            <section className="evaluation-comparison-insights" aria-label="Observaciones destacadas">
                                <div><span className="evaluation-eyebrow">Resumen interpretativo</span><h3>Observaciones destacadas</h3><p>Observaciones descriptivas calculadas a partir de las métricas de A y B. No establecen una superioridad global.</p></div>
                                <div>{insights.map((insight) => <article key={insight.key} className={`side-${insight.side}`}><span aria-hidden="true">{insight.side.toUpperCase()}</span><p>{insight.text}</p></article>)}</div>
                            </section>
                        )}

                        <div className="evaluation-comparison-metrics" role="table" aria-label="Comparación de métricas">
                            <div className="evaluation-comparison-metric header" role="row">
                                <span role="columnheader">Métrica</span>
                                <span role="columnheader">A</span>
                                <span role="columnheader">B</span>
                                <span role="columnheader">Δ B − A</span>
                            </div>
                            {metrics.map((entry) => (
                                <div key={entry.key} className="evaluation-comparison-metric" role="row">
                                    <div role="cell" className="evaluation-comparison-metric-name">
                                        <div><strong>{entry.label}</strong></div>
                                        <div className="evaluation-comparison-mini-bars" aria-hidden="true"><span className="side-a"><i style={{ width: metricBarWidth(entry.left) }} /></span><span className="side-b"><i style={{ width: metricBarWidth(entry.right) }} /></span></div>
                                    </div>
                                    <span role="cell">{formatMetric(entry.left, entry.format)}</span>
                                    <span role="cell">{formatMetric(entry.right, entry.format)}</span>
                                    <span role="cell" className="evaluation-comparison-delta">{formatDelta(entry.delta)}</span>
                                </div>
                            ))}
                        </div>
                    </section>
                )}
            />
        </section>
    )
}

export default EvaluationComparisonView
