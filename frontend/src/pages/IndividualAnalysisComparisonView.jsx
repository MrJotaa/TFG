import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { getDatasetById } from '../api/datasets'
import {
    getStoredIndividualAnalysis,
    saveIndividualAnalysis,
} from '../api/individualAnalysisHistory'
import IndividualAnalysisComparisonExplorer from '../components/individualAnalysisComparison/IndividualAnalysisComparisonExplorer'
import { useExperiment } from '../context/ExperimentContext'
import { useNavigationGuard } from '../context/NavigationGuardContext'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import {
    buildIndividualAnalysisComparison,
    formatComparisonMagnitude,
    individualAnalysisComparisonSummary,
} from '../utils/individualAnalysisComparison'
import { resolveSaveState } from '../utils/individualAnalysisFlow'
import { formatAnalysisProbability, formatSignedPercentagePoints } from '../utils/individualAnalysis'
import { formatCreatedAgo } from '../utils/evaluationHistory'

function analysisPersistenceLabel(reference, saveStatus) {
    if (!reference?.temporary) {
        return reference?.item?.saved_at
            ? `guardado ${formatCreatedAgo(reference.item.saved_at)}`
            : 'guardado'
    }
    if (saveStatus === 'equivalent') return 'temporal · equivalente guardado'
    return 'temporal · no guardado'
}

function AnalysisIdentity({ side, reference, attackCatalog, saveStatus }) {
    const item = reference.item
    return (
        <article className={`individual-comparison-identity side-${side.toLowerCase()}`}>
            <div className="individual-comparison-identity-topline">
                <span>{side}</span>
                <small>{analysisPersistenceLabel(reference, saveStatus)}</small>
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

function ClassificationTopThree({ title, predictions = [] }) {
    const topThree = predictions.slice(0, 3)

    return (
        <section className="individual-comparison-topk-column">
            <h4>{title}</h4>
            {topThree.length > 0 ? (
                <ol>
                    {topThree.map((prediction, index) => (
                        <li key={`${prediction.label}-${index}`}>
                            <span className="individual-comparison-topk-rank">{index + 1}</span>
                            <span title={prediction.label}>{prediction.label}</span>
                            <strong>{formatAnalysisProbability(prediction.probability)}</strong>
                        </li>
                    ))}
                </ol>
            ) : (
                <p>Top-3 no disponible.</p>
            )}
        </section>
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
            <div className="individual-comparison-topk">
                <ClassificationTopThree
                    title="Top-3 original"
                    predictions={summary.originalTopK}
                />
                <ClassificationTopThree
                    title="Top-3 tras el ataque"
                    predictions={summary.adversarialTopK}
                />
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

function PersistenceAction({ side, reference, status, saving, onSave }) {
    if (!reference.temporary) {
        return (
            <div className={`individual-comparison-save-state side-${side.toLowerCase()}`}>
                <span>{side}</span>
                <div>
                    <strong>Guardado</strong>
                    <small>Este lado ya está conservado en el historial.</small>
                </div>
            </div>
        )
    }

    const equivalent = status === 'equivalent'
    return (
        <div className={`individual-comparison-save-state side-${side.toLowerCase()} temporary`}>
            <span>{side}</span>
            <div>
                <strong>{equivalent ? 'Equivalente ya guardado' : 'Resultado temporal'}</strong>
                <small>
                    {equivalent
                        ? 'Existe la misma configuración en historial, pero esta ejecución concreta sigue siendo temporal.'
                        : 'Puedes compararlo ahora sin guardarlo o conservar esta ejecución indefinidamente.'}
                </small>
            </div>
            <button
                type="button"
                className="secondary-action"
                onClick={onSave}
                disabled={saving || equivalent}
            >
                {saving ? 'Guardando…' : equivalent ? 'Ya existe' : `Guardar ${side}`}
            </button>
        </div>
    )
}

function buildSavedReference(stored, dataset, image) {
    return {
        item: stored.item,
        analysis: stored.analysis,
        dataset,
        image,
        temporary: false,
        resultId: stored.analysis?.result_id ?? stored.item?.source_result_id ?? null,
    }
}

function IndividualAnalysisComparisonView() {
    const navigate = useNavigate()
    const location = useLocation()
    const [searchParams] = useSearchParams()
    const { attackCatalog } = useExperiment()
    const { setNavigationProtection, clearNavigationProtection } = useNavigationGuard()
    const [state, setState] = useState({ status: 'loading' })
    const [savingSide, setSavingSide] = useState(null)
    const [saveStatuses, setSaveStatuses] = useState({ A: null, B: null })

    const leftId = searchParams.get('a')
    const rightId = searchParams.get('b')
    const transientPair = location.state?.individualComparisonPair ?? null

    useEffect(() => {
        let cancelled = false

        async function load() {
            try {
                setSaveStatuses({ A: null, B: null })

                if (transientPair) {
                    const left = transientPair.left
                    const right = transientPair.right
                    if (!left?.item || !right?.item || !left?.analysis || !right?.analysis) {
                        throw new Error('La comparación temporal ya no contiene los dos resultados necesarios.')
                    }
                    const comparison = buildIndividualAnalysisComparison(left.item, right.item)
                    if (!comparison.compatible) throw new Error(comparison.reason)

                    const dataset = transientPair.dataset ?? left.dataset ?? right.dataset
                    const image = transientPair.image ?? left.image ?? right.image
                    if (!dataset || !image) {
                        throw new Error('No se pudo recuperar el contexto visual de la imagen comparada.')
                    }

                    if (!cancelled) {
                        setState({
                            status: 'ready',
                            left,
                            right,
                            comparison,
                            dataset,
                            image,
                        })
                    }
                    return
                }

                if (!leftId || !rightId || leftId === rightId) {
                    throw new Error('Selecciona dos análisis individuales diferentes desde el historial.')
                }

                setState({ status: 'loading' })
                const [leftStored, rightStored] = await Promise.all([
                    getStoredIndividualAnalysis(leftId),
                    getStoredIndividualAnalysis(rightId),
                ])
                const comparison = buildIndividualAnalysisComparison(
                    leftStored.item,
                    rightStored.item,
                )
                if (!comparison.compatible) throw new Error(comparison.reason)

                const dataset = await getDatasetById(leftStored.item.dataset_id)
                const image = dataset?.images?.find(
                    (candidate) => candidate.id === leftStored.item.image_id,
                )
                if (!image) {
                    throw new Error('La imagen original asociada ya no está disponible en el dataset.')
                }

                if (!cancelled) {
                    setState({
                        status: 'ready',
                        left: buildSavedReference(leftStored, dataset, image),
                        right: buildSavedReference(rightStored, dataset, image),
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
    }, [leftId, rightId, transientPair])

    const summaries = useMemo(() => {
        if (state.status !== 'ready') return null
        return {
            left: individualAnalysisComparisonSummary(
                state.left.item.task_type,
                state.left.analysis,
            ),
            right: individualAnalysisComparisonSummary(
                state.right.item.task_type,
                state.right.analysis,
            ),
        }
    }, [state])

    const hasTemporaryWork = state.status === 'ready'
        && (state.left.temporary || state.right.temporary)
    const busy = Boolean(savingSide)

    useEffect(() => {
        setNavigationProtection({ hasTemporaryWork, busy })
        return () => clearNavigationProtection()
    }, [hasTemporaryWork, busy, setNavigationProtection, clearNavigationProtection])

    async function handleSave(side) {
        if (state.status !== 'ready') return
        const key = side === 'A' ? 'left' : 'right'
        const reference = state[key]
        if (!reference?.temporary || !reference.resultId) return

        try {
            setSavingSide(side)
            setState((current) => ({ ...current, saveError: '' }))
            const item = await saveIndividualAnalysis({
                taskType: reference.item.task_type,
                resultId: reference.resultId,
            })
            const status = resolveSaveState(reference.resultId, item)
            setSaveStatuses((current) => ({ ...current, [side]: status }))

            if (status !== 'saved') return

            const savedReference = {
                ...reference,
                item,
                temporary: false,
            }
            const other = side === 'A' ? state.right : state.left
            setState((current) => ({
                ...current,
                [key]: savedReference,
            }))

            if (!other.temporary) {
                const leftAnalysisId = side === 'A'
                    ? item.analysis_id
                    : state.left.item.analysis_id
                const rightAnalysisId = side === 'B'
                    ? item.analysis_id
                    : state.right.item.analysis_id
                const search = new URLSearchParams({
                    a: leftAnalysisId,
                    b: rightAnalysisId,
                })
                clearNavigationProtection()
                navigate(`/individual-analyses/compare?${search.toString()}`, {
                    replace: true,
                })
            }
        } catch (error) {
            setState((current) => ({
                ...current,
                saveError: error.message || `No se pudo guardar el análisis ${side}.`,
            }))
        } finally {
            setSavingSide(null)
        }
    }

    if (state.status === 'loading') {
        return (
            <section className="individual-comparison-page">
                <div className="individual-comparison-loading">
                    <strong>Preparando comparación visual…</strong>
                    <span>Recuperando los análisis disponibles sin volver a ejecutar ataques.</span>
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
        left,
        right,
        comparison,
        dataset,
        image,
        saveError,
    } = state
    const leftItem = left.item
    const rightItem = right.item
    const leftAnalysis = left.analysis
    const rightAnalysis = right.analysis
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
                <AnalysisIdentity
                    side="A"
                    reference={left}
                    attackCatalog={attackCatalog}
                    saveStatus={saveStatuses.A}
                />
                <span className="individual-comparison-versus">vs</span>
                <AnalysisIdentity
                    side="B"
                    reference={right}
                    attackCatalog={attackCatalog}
                    saveStatus={saveStatuses.B}
                />
            </div>

            {(left.temporary || right.temporary) && (
                <section className="individual-comparison-save-strip" aria-label="Persistencia de la comparación">
                    <PersistenceAction
                        side="A"
                        reference={left}
                        status={saveStatuses.A}
                        saving={savingSide === 'A'}
                        onSave={() => handleSave('A')}
                    />
                    <PersistenceAction
                        side="B"
                        reference={right}
                        status={saveStatuses.B}
                        saving={savingSide === 'B'}
                        onSave={() => handleSave('B')}
                    />
                </section>
            )}
            {saveError && <p className="error individual-comparison-save-error">{saveError}</p>}

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
                    {hasTemporaryWork && (
                        <p>
                            <strong>Persistencia:</strong> al menos uno de los lados sigue siendo
                            temporal. Puedes compararlo sin guardarlo, pero se perderá al abandonar
                            la vista si no decides conservarlo.
                        </p>
                    )}
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
                        <strong>L∞ {formatComparisonMagnitude(summaries.left.perturbationLinf)}</strong>
                        <small>L2 {formatComparisonMagnitude(summaries.left.perturbationL2)}</small>
                    </div>
                    <div className="side-b">
                        <span>B · perturbación</span>
                        <strong>L∞ {formatComparisonMagnitude(summaries.right.perturbationLinf)}</strong>
                        <small>L2 {formatComparisonMagnitude(summaries.right.perturbationL2)}</small>
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
