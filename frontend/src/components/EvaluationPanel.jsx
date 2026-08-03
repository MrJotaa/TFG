import { useMemo, useReducer, useRef, useState } from 'react'
import {
    evaluateDataset,
    generateEvaluationGradCam,
} from '../api/evaluation'
import EvaluationClassMetrics from './evaluation/EvaluationClassMetrics'
import EvaluationConfusionMatrices from './evaluation/EvaluationConfusionMatrices'
import EvaluationCurves from './evaluation/EvaluationCurves'
import EvaluationImages from './evaluation/EvaluationImages'
import EvaluationMetricGuide from './evaluation/EvaluationMetricGuide'
import EvaluationOverview from './evaluation/EvaluationOverview'
import EvaluationTabs from './evaluation/EvaluationTabs'
import {
    canRequestEvaluationGradCam,
    evaluationGradCamReducer,
    getEvaluationGradCamEntry,
    getEvaluationGradCamErrorMessage,
    getEvaluationGradCamKey,
    isEvaluationGradCamResponse,
} from '../utils/evaluationGradCamState'
import { CLASSIFICATION_METRIC_GUIDES } from '../utils/evaluationMetricGuides'

const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
}

const DEFAULT_ATTACK_PARAMETERS = {
    epsilon: 0.03,
    alpha: 0.01,
    steps: 10,
}

function EvaluationPanel({
    selectedDataset,
    selectedAttack = 'fgsm',
    attackParameters = DEFAULT_ATTACK_PARAMETERS,
}) {
    const [evaluationState, setEvaluationState] = useState(null)
    const [activeTab, setActiveTab] = useState('overview')
    const [loading, setLoading] = useState(false)
    const [errorState, setErrorState] = useState(null)
    const [gradCamStates, dispatchGradCam] = useReducer(
        evaluationGradCamReducer,
        {},
    )
    const gradCamRequests = useRef(new Set())
    const gradCamRun = useRef(0)

    const epsilon = attackParameters.epsilon
    const alpha = attackParameters.alpha
    const steps = attackParameters.steps

    const contextKey = [
        selectedDataset?.id || '',
        selectedAttack,
        epsilon,
        alpha,
        steps,
    ].join(':')

    const evaluation =
        evaluationState?.contextKey === contextKey ? evaluationState.data : null
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

    async function handleEvaluate() {
        if (!selectedDataset) {
            setErrorState({
                contextKey,
                message: 'Debes seleccionar un dataset.',
            })
            return
        }

        try {
            setLoading(true)
            setErrorState(null)
            gradCamRun.current += 1
            gradCamRequests.current.clear()
            dispatchGradCam({ type: 'reset' })

            const result = await evaluateDataset({
                datasetId: selectedDataset.id,
                attackType: selectedAttack,
                epsilon,
                alpha,
                steps,
            })

            setEvaluationState({ contextKey, data: result })
            setActiveTab('overview')
        } catch (err) {
            setErrorState({ contextKey, message: err.message })
        } finally {
            setLoading(false)
        }
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
                throw new Error('Invalid Grad-CAM response')
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

    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack

    if (!selectedDataset) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Configuración incompleta</span>
                <h3>No hay dataset seleccionado</h3>
                <p>
                    Vuelve al dashboard y selecciona un dataset antes de ejecutar la
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
                        <h3>Analiza el dataset completo con {attackName}</h3>
                        <p>
                            Se compararán las predicciones originales y adversariales de{' '}
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

            {loading && (
                <section className="evaluation-loading-card" role="status" aria-live="polite">
                    <div className="evaluation-loader" aria-hidden="true" />
                    <div>
                        <strong>Evaluando imágenes y calculando métricas</strong>
                        <p>
                            El tiempo depende del número de imágenes, el ataque y el modelo
                            activo.
                        </p>
                    </div>
                </section>
            )}

            {evaluation && !loading && (
                <section className="evaluation-results-shell" aria-label="Resultados de evaluación">
                    <p className="evaluation-sr-only" role="status" aria-live="polite">
                        Evaluación completada. Se muestran los resultados del dataset.
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
                                Ejecuta de nuevo la misma configuración de dataset,
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
