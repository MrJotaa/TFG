import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { getDatasetById } from '../api/datasets'
import { getStoredIndividualAnalysis } from '../api/individualAnalysisHistory'
import IndividualAnalysisComparisonExplorer from '../components/individualAnalysisComparison/IndividualAnalysisComparisonExplorer'
import { useExperiment } from '../context/ExperimentContext'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import {
    buildIndividualAnalysisComparison,
    individualAnalysisComparisonSummary,
} from '../utils/individualAnalysisComparison'
import { formatAnalysisProbability, formatSignedPercentagePoints } from '../utils/individualAnalysis'
import { formatCreatedAgo } from '../utils/evaluationHistory'

function formatMagnitude(value) {
    const numeric = Number(value)
    if (!Number.isFinite(numeric)) return '—'
    if (numeric === 0) return '0'
    if (Math.abs(numeric) < 0.001) return numeric.toExponential(2)
    return numeric.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

function AnalysisIdentity({ side, item, attackCatalog }) {
    return (
        <article className={`individual-comparison-identity side-${side.toLowerCase()}`}>
            <div className="individual-comparison-identity-topline">
                <span>{side}</span>
                <small>guardado {formatCreatedAgo(item.saved_at)}</small>
            </div>
            <strong>{item.model_name}</strong>
            <div>
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

function ClassificationSummary({ side, summary }) {
    return (
        <article className={`individual-comparison-result-card side-${side.toLowerCase()}`}>
            <header>
                <span className="individual-comparison-side-token">{side}</span>
                <strong>{summary.primaryCaption}</strong>
            </header>
            <p className="individual-comparison-result-flow">{summary.primaryLabel}</p>
            <div className="individual-comparison-result-metrics">
                <div>
                    <span>Confianza original</span>
                    <strong>{formatAnalysisProbability(summary.originalConfidence)}</strong>
                </div>
                <div>
                    <span>Tras ataque</span>
                    <strong>{formatAnalysisProbability(summary.adversarialConfidence)}</strong>
                </div>
                <div>
                    <span>Δ confianza</span>
                    <strong>{formatSignedPercentagePoints(summary.confidenceDifference)}</strong>
                </div>
            </div>
        </article>
    )
}

function DetectionSummary({ side, summary }) {
    return (
        <article className={`individual-comparison-result-card side-${side.toLowerCase()}`}>
            <header>
                <span className="individual-comparison-side-token">{side}</span>
                <strong>{summary.attackSuccess ? 'Impacto detectado' : 'Detecciones preservadas'}</strong>
            </header>
            <p className="individual-comparison-result-flow">
                {summary.primaryLabel} <small>{summary.primaryCaption}</small>
            </p>
            <div className="individual-comparison-result-metrics">
                <div>
                    <span>Confianza media original</span>
                    <strong>{formatAnalysisProbability(summary.originalConfidence)}</strong>
                </div>
                <div>
                    <span>Tras ataque</span>
                    <strong>{formatAnalysisProbability(summary.adversarialConfidence)}</strong>
                </div>
                <div>
                    <span>Δ confianza</span>
                    <strong>{formatSignedPercentagePoints(summary.confidenceDifference)}</strong>
                </div>
            </div>
        </article>
    )
}

function IndividualAnalysisComparisonView() {
    const navigate = useNavigate()
    const [searchParams] = useSearchParams()
    const { attackCatalog } = useExperiment()
    const [state, setState] = useState({ status: 'loading' })

    const leftId = searchParams.get('a')
    const rightId = searchParams.get('b')

    useEffect(() => {
        let cancelled = false

        async function load() {
            if (!leftId || !rightId || leftId === rightId) {
                setState({
                    status: 'error',
                    message: 'Selecciona dos análisis individuales diferentes desde el historial.',
                })
                return
            }

            try {
                setState({ status: 'loading' })
                const [leftStored, rightStored] = await Promise.all([
                    getStoredIndividualAnalysis(leftId),
                    getStoredIndividualAnalysis(rightId),
                ])
                const comparison = buildIndividualAnalysisComparison(
                    leftStored.item,
                    rightStored.item,
                )
                if (!comparison.compatible) {
                    throw new Error(comparison.reason)
                }

                const dataset = await getDatasetById(leftStored.item.dataset_id)
                const image = dataset?.images?.find(
                    (candidate) => candidate.id === leftStored.item.image_id,
                )
                if (!image) {
                    throw new Error(
                        'La imagen original asociada ya no está disponible en el dataset.',
                    )
                }

                if (!cancelled) {
                    setState({
                        status: 'ready',
                        leftItem: leftStored.item,
                        rightItem: rightStored.item,
                        leftAnalysis: leftStored.analysis,
                        rightAnalysis: rightStored.analysis,
                        comparison,
                        dataset,
                        image,
                    })
                }
            } catch (error) {
                if (!cancelled) {
                    setState({
                        status: 'error',
                        message: error.message || 'No se pudo preparar la comparación.',
                    })
                }
            }
        }

        load()
        return () => {
            cancelled = true
        }
    }, [leftId, rightId])

    const summaries = useMemo(() => {
        if (state.status !== 'ready') return null
        return {
            left: individualAnalysisComparisonSummary(
                state.leftItem.task_type,
                state.leftAnalysis,
            ),
            right: individualAnalysisComparisonSummary(
                state.rightItem.task_type,
                state.rightAnalysis,
            ),
        }
    }, [state])

    if (state.status === 'loading') {
        return (
            <section className="individual-comparison-page">
                <div className="individual-comparison-loading">
                    <strong>Preparando comparación visual…</strong>
                    <span>Recuperando los dos análisis guardados sin volver a ejecutar ataques.</span>
                </div>
            </section>
        )
    }

    if (state.status === 'error') {
        return (
            <section className="individual-comparison-page">
                <div className="individual-comparison-error">
                    <span className="evaluation-eyebrow">Comparación individual</span>
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
        leftAnalysis,
        rightAnalysis,
        comparison,
        dataset,
        image,
    } = state
    const taskType = leftItem.task_type
    const taskLabel = taskType === 'detection' ? 'detección' : 'clasificación'
    const Summary = taskType === 'detection' ? DetectionSummary : ClassificationSummary
    const leftAttackName = getAttackName(leftItem.attack_type, attackCatalog)
    const rightAttackName = getAttackName(rightItem.attack_type, attackCatalog)
    const leftVisualLabel = comparison.kind === 'model'
        ? leftItem.model_name
        : leftAttackName
    const rightVisualLabel = comparison.kind === 'model'
        ? rightItem.model_name
        : rightAttackName

    return (
        <section className="individual-comparison-page">
            <header className="individual-comparison-hero">
                <div>
                    <span className="evaluation-eyebrow">
                        Comparación individual · {taskLabel}
                    </span>
                    <h1>Una imagen, dos condiciones experimentales</h1>
                    <p>
                        Comparación cualitativa sobre el mismo ejemplo. Sirve para inspeccionar
                        cómo cambia el resultado; las conclusiones globales siguen correspondiendo
                        a las evaluaciones sobre el dataset completo.
                    </p>
                </div>
                <div className="individual-comparison-context">
                    <strong>{image.filename}</strong>
                    <span>{dataset.name} · revisión {leftItem.dataset_revision}</span>
                    <small>{comparison.label} · {comparison.description}</small>
                </div>
            </header>

            <div className="individual-comparison-identities" aria-label="Análisis comparados">
                <AnalysisIdentity side="A" item={leftItem} attackCatalog={attackCatalog} />
                <span className="individual-comparison-versus">vs</span>
                <AnalysisIdentity side="B" item={rightItem} attackCatalog={attackCatalog} />
            </div>

            <details className="individual-comparison-guide">
                <summary>
                    <span className="individual-comparison-guide-icon" aria-hidden="true">i</span>
                    <span>
                        <strong>Cómo leer esta comparación</strong>
                        <small>Qué puede concluirse de una sola imagen</small>
                    </span>
                    <span aria-hidden="true">⌄</span>
                </summary>
                <div>
                    <p>
                        A y B se ejecutaron sobre la misma imagen y solo cambia una variable
                        experimental. La vista permite estudiar predicciones, detecciones,
                        perturbaciones y explicabilidad persistida sin recalcular el ataque.
                    </p>
                    <p>
                        <strong>Importante:</strong> un caso individual explica comportamiento local,
                        pero no demuestra por sí solo que un modelo o ataque sea globalmente superior.
                        Para eso utiliza la comparación de evaluaciones del dataset.
                    </p>
                </div>
            </details>

            <section className="individual-comparison-result-overview" aria-labelledby="individual-result-overview-title">
                <header>
                    <span className="evaluation-eyebrow">Resultado del caso</span>
                    <h2 id="individual-result-overview-title">Qué ocurrió bajo A y B</h2>
                </header>
                <div className="individual-comparison-result-grid">
                    <Summary side="A" summary={summaries.left} />
                    <Summary side="B" summary={summaries.right} />
                </div>
                <div className="individual-comparison-perturbation-strip">
                    <div className="side-a">
                        <span>A · perturbación</span>
                        <strong>L∞ {formatMagnitude(summaries.left.perturbationLinf)}</strong>
                        <small>L2 {formatMagnitude(summaries.left.perturbationL2)}</small>
                    </div>
                    <div className="side-b">
                        <span>B · perturbación</span>
                        <strong>L∞ {formatMagnitude(summaries.right.perturbationLinf)}</strong>
                        <small>L2 {formatMagnitude(summaries.right.perturbationL2)}</small>
                    </div>
                </div>
            </section>

            <IndividualAnalysisComparisonExplorer
                taskType={taskType}
                image={image}
                leftAnalysis={leftAnalysis}
                rightAnalysis={rightAnalysis}
                leftLabel={leftVisualLabel}
                rightLabel={rightVisualLabel}
                comparisonKind={comparison.kind}
            />
        </section>
    )
}

export default IndividualAnalysisComparisonView
