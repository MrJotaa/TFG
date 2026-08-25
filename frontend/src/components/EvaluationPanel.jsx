import { useMemo, useReducer, useRef, useState } from 'react'
import {
    evaluateDataset,
    generateEvaluationGradCam,
} from '../api/evaluation'
import {
    findCompatibleEvaluation,
    getStoredEvaluation,
} from '../api/evaluationHistory'
import EvaluationClassMetrics from './evaluation/EvaluationClassMetrics'
import EvaluationConfusionMatrices from './evaluation/EvaluationConfusionMatrices'
import EvaluationCurves from './evaluation/EvaluationCurves'
import EvaluationImages from './evaluation/EvaluationImages'
import EvaluationMetricGuide from './evaluation/EvaluationMetricGuide'
import EvaluationOverview from './evaluation/EvaluationOverview'
import EvaluationProgress from './evaluation/EvaluationProgress'
import EvaluationTabs from './evaluation/EvaluationTabs'
import CompatibleEvaluationNotice from './evaluation/CompatibleEvaluationNotice'
import {
    canRequestEvaluationGradCam,
    evaluationGradCamReducer,
    getEvaluationGradCamEntry,
    getEvaluationGradCamErrorMessage,
    getEvaluationGradCamKey,
    isEvaluationGradCamResponse,
} from '../utils/evaluationGradCamState'
import { CLASSIFICATION_METRIC_GUIDES } from '../utils/evaluationMetricGuides'
import { attackParametersKey } from '../utils/attackCatalog'
import { getAttackName } from '../utils/attackFormatting'
import { isRequestCancelled } from '../api/client'

function EvaluationPanel({
    selectedDataset,
    selectedAttack = 'fgsm',
    attackParameters = {},
    attackCatalog = [],
    recoveredEvaluation = null,
    onStartNewEvaluation = () => {},
}) {
    const [evaluationState, setEvaluationState] = useState(null)
    const [activeTab, setActiveTab] = useState('overview')
    const [loading, setLoading] = useState(false)
    const [progress, setProgress] = useState(null)
    const [errorState, setErrorState] = useState(null)
    const [reuseCandidate, setReuseCandidate] = useState(null)
    const [gradCamStates, dispatchGradCam] = useReducer(
        evaluationGradCamReducer,
        {},
    )
    const gradCamRequests = useRef(new Set())
    const gradCamRun = useRef(0)
    const evaluationAbortController = useRef(null)

    const contextKey = [
        selectedDataset?.id || '',
        selectedAttack,
        attackParametersKey(attackParameters),
    ].join(':')

    const evaluation = recoveredEvaluation?.data ?? (
        evaluationState?.contextKey === contextKey ? evaluationState.data : null
    )
    const error = errorState?.contextKey === contextKey ? errorState.message : ''

    const tabs = useMemo(() => {
        const perClassCount = evaluation?.advanced_metrics?.supervised?.per_class?.length
        const imageCount = evaluation?.results?.length

        return [
            { id: 'overview', label: 'Resumen' },
            { id: 'classes', label: 'Por clase', badge: perClassCount },
            { id: 'matrices', label: 'Matrices' },
            { id: 'curves', label: 'Curvas' },
            { id: 'images', label: 'Imágenes', badge: imageCount },
        ]
    }, [evaluation])

    async function runEvaluation(forceRecompute) {
        setReuseCandidate(null)
        setProgress(null)
        gradCamRun.current += 1
        gradCamRequests.current.clear()
        dispatchGradCam({ type: 'reset' })
        const controller = new AbortController()
        evaluationAbortController.current = controller

        try {
            const result = await evaluateDataset({
                datasetId: selectedDataset.id,
                attackType: selectedAttack,
                parameters: attackParameters,
                forceRecompute,
                onProgress: setProgress,
                signal: controller.signal,
            })
            onStartNewEvaluation()
            setEvaluationState({ contextKey, data: result })
            setActiveTab('overview')
        } finally {
            if (evaluationAbortController.current === controller) {
                evaluationAbortController.current = null
            }
        }
    }

    async function handleEvaluate() {
        if (!selectedDataset) {
            setErrorState({
                contextKey,
                message: 'Debes seleccionar un conjunto de datos.',
            })
            return
        }

        try {
            setLoading(true)
            setErrorState(null)
            const compatibility = await findCompatibleEvaluation({
                taskType: 'classification',
                datasetId: selectedDataset.id,
                attackType: selectedAttack,
                parameters: attackParameters,
            })
            if (compatibility.match) {
                setReuseCandidate(compatibility.match)
                return
            }
            await runEvaluation(false)
        } catch (err) {
            if (!isRequestCancelled(err)) {
                setErrorState({ contextKey, message: err.message })
            }
        } finally {
            setLoading(false)
            setProgress(null)
        }
    }

    async function handleReuseCandidate() {
        try {
            setLoading(true)
            setErrorState(null)
            const stored = await getStoredEvaluation(reuseCandidate.result_id)
            if (stored.task_type !== 'classification') {
                throw new Error('El resultado compatible no es de clasificación.')
            }
            onStartNewEvaluation()
            setEvaluationState({ contextKey, data: stored.evaluation })
            setReuseCandidate(null)
            setActiveTab('overview')
        } catch (err) {
            if (!isRequestCancelled(err)) {
                setErrorState({ contextKey, message: err.message })
            }
        } finally {
            setLoading(false)
            setProgress(null)
        }
    }

    async function handleRunAgain() {
        try {
            setLoading(true)
            setErrorState(null)
            await runEvaluation(true)
        } catch (err) {
            if (!isRequestCancelled(err)) {
                setErrorState({ contextKey, message: err.message })
            }
        } finally {
            setLoading(false)
            setProgress(null)
        }
    }

    function handleCancelEvaluation() {
        evaluationAbortController.current?.abort()
    }

    async function handleGenerateGradCam(evaluationId, imageId) {
        const key = getEvaluationGradCamKey(evaluationId, imageId)
        const requestRun = gradCamRun.current
        const currentEntry = getEvaluationGradCamEntry(gradCamStates, key)

        if (
            !canRequestEvaluationGradCam(
                currentEntry,
                gradCamRequests.current.has(key),
            )
        ) {
            return
        }

        gradCamRequests.current.add(key)
        dispatchGradCam({ type: 'loading', key })

        try {
            const response = await generateEvaluationGradCam(
                evaluationId,
                imageId,
            )

            if (
                !isEvaluationGradCamResponse(
                    response,
                    evaluationId,
                    imageId,
                )
            ) {
                throw new Error('Respuesta de Grad-CAM no válida')
            }

            if (requestRun !== gradCamRun.current) {
                return
            }

            dispatchGradCam({
                type: 'success',
                key,
                data: response,
            })
        } catch (err) {
            if (requestRun !== gradCamRun.current) {
                return
            }

            dispatchGradCam({
                type: 'error',
                key,
                error: getEvaluationGradCamErrorMessage(err),
            })
        } finally {
            if (requestRun === gradCamRun.current) {
                gradCamRequests.current.delete(key)
            }
        }
    }

    const attackName = getAttackName(selectedAttack, attackCatalog)

    if (!selectedDataset) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Configuración incompleta</span>
                <h3>No hay conjunto de datos seleccionado</h3>
                <p>
                    Vuelve al panel principal y selecciona un conjunto de datos antes de ejecutar la
                    evaluación.
                </p>
            </div>
        )
    }

    return (
        <div className="evaluation-panel evaluation-dashboard">
            {!evaluation && !loading && (
                <section className="evaluation-run-card modern">
                    <div>
                        <span className="evaluation-eyebrow">Evaluación de robustez</span>
                        <h3>Analiza el conjunto de datos completo con {attackName}</h3>
                        <p>
                            Se compararán las predicciones originales y las obtenidas tras el ataque de{' '}
                            <strong>{selectedDataset.image_count}</strong> imágenes.
                        </p>
                    </div>

                    <button
                        type="button"
                        className="primary-action evaluation-run-button"
                        onClick={handleEvaluate}
                    >
                        Ejecutar evaluación
                    </button>
                </section>
            )}

            {error && (
                <div className="evaluation-error-banner" role="alert">
                    <strong>No se pudo completar la evaluación.</strong>
                    <span>{error}</span>
                </div>
            )}

            {reuseCandidate && !loading && (
                <CompatibleEvaluationNotice
                    item={reuseCandidate}
                    onReuse={handleReuseCandidate}
                    onRunAgain={handleRunAgain}
                    onRefresh={handleEvaluate}
                    onDismiss={() => setReuseCandidate(null)}
                />
            )}

            {loading && (
                <EvaluationProgress
                    progress={progress}
                    title="Evaluando imágenes y calculando métricas"
                    pendingMessage="Preparando la evaluación del conjunto de datos..."
                    onCancel={handleCancelEvaluation}
                />
            )}

            {evaluation && !loading && (
                <section className="evaluation-results-shell" aria-label="Resultados de evaluación">
                    <p className="evaluation-sr-only" role="status" aria-live="polite">
                        Evaluación completada. Se muestran los resultados del conjunto de datos.
                    </p>
                    <EvaluationTabs
                        tabs={tabs}
                        activeTab={activeTab}
                        onChange={setActiveTab}
                    />
                    <EvaluationMetricGuide
                        key={activeTab}
                        guide={CLASSIFICATION_METRIC_GUIDES[activeTab]}
                    />

                    <div
                        id={`evaluation-panel-${activeTab}`}
                        className="evaluation-tab-panel"
                        role="tabpanel"
                        aria-labelledby={`evaluation-tab-${activeTab}`}
                        tabIndex={0}
                    >
                        {activeTab === 'overview' && (
                            <EvaluationOverview evaluation={evaluation} />
                        )}

                        {activeTab === 'classes' && (
                            <EvaluationClassMetrics evaluation={evaluation} />
                        )}

                        {activeTab === 'matrices' && (
                            <EvaluationConfusionMatrices evaluation={evaluation} />
                        )}

                        {activeTab === 'curves' && (
                            <EvaluationCurves evaluation={evaluation} />
                        )}

                        {activeTab === 'images' && (
                            <EvaluationImages
                                evaluation={evaluation}
                                gradCamStates={gradCamStates}
                                onGenerateGradCam={handleGenerateGradCam}
                            />
                        )}
                    </div>

                    <footer className="evaluation-rerun-footer">
                        <div>
                            <strong>¿Necesitas actualizar los resultados?</strong>
                            <span>
                                Ejecuta de nuevo la misma configuración de conjunto de datos,
                                modelo y ataque.
                            </span>
                        </div>

                        <button
                            type="button"
                            className="secondary-action evaluation-rerun-button"
                            onClick={handleEvaluate}
                        >
                            Repetir evaluación
                        </button>
                    </footer>
                </section>
            )}
        </div>
    )
}

export default EvaluationPanel
