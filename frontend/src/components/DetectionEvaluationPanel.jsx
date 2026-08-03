import { useMemo, useReducer, useRef, useState } from 'react'
import {
    evaluateDetectionDataset,
    generateDetectionEvaluationEigenCam,
} from '../api/detectionEvaluation'
import DetectionClassMetrics from './detectionEvaluation/DetectionClassMetrics'
import DetectionCocoMetrics from './detectionEvaluation/DetectionCocoMetrics'
import DetectionImageResults from './detectionEvaluation/DetectionImageResults'
import DetectionObjectAnalysis from './detectionEvaluation/DetectionObjectAnalysis'
import DetectionOverview from './detectionEvaluation/DetectionOverview'
import DetectionPrecisionRecall from './detectionEvaluation/DetectionPrecisionRecall'
import DetectionTideAnalysis from './detectionEvaluation/DetectionTideAnalysis'
import EvaluationMetricGuide from './evaluation/EvaluationMetricGuide'
import EvaluationTabs from './evaluation/EvaluationTabs'
import {
    canRequestDetectionEigenCam,
    detectionEigenCamReducer,
    getDetectionEigenCamEntry,
    getDetectionEigenCamErrorMessage,
    getDetectionEigenCamKey,
    isDetectionEigenCamResponse,
} from '../utils/detectionEvaluationEigenCamState'
import { DETECTION_METRIC_GUIDES } from '../utils/evaluationMetricGuides'

const ATTACK_NAMES = {
    random_noise: 'Random Noise',
    yolo_confidence: 'YOLO Confidence',
}

const SUPPORTED_DETECTION_ATTACKS = ['random_noise', 'yolo_confidence']
const IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

function DetectionEvaluationPanel({
    selectedDataset,
    selectedAttack,
    attackParameters,
}) {
    const [evaluationState, setEvaluationState] = useState(null)
    const [activeTab, setActiveTab] = useState('overview')
    const [loading, setLoading] = useState(false)
    const [errorState, setErrorState] = useState(null)
    const [configuration, setConfiguration] = useState({
        confidence_threshold: 0.25,
        iou_threshold: 0.5,
        include_object_level_details: false,
    })
    const [eigenCamStates, dispatchEigenCam] = useReducer(
        detectionEigenCamReducer,
        {},
    )
    const eigenCamRequests = useRef(new Set())
    const eigenCamRun = useRef(0)

    const isSupportedAttack = SUPPORTED_DETECTION_ATTACKS.includes(selectedAttack)
    const isConfigurationValid =
        Number.isFinite(configuration.confidence_threshold) &&
        configuration.confidence_threshold >= 0.001 &&
        configuration.confidence_threshold <= 1
    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack
    const contextKey = [
        selectedDataset?.id ?? '',
        selectedAttack,
        attackParameters.epsilon,
        attackParameters.alpha,
        attackParameters.steps,
        configuration.confidence_threshold,
        configuration.iou_threshold,
        configuration.include_object_level_details,
    ].join(':')

    const evaluation =
        evaluationState?.contextKey === contextKey ? evaluationState.data : null
    const error = errorState?.contextKey === contextKey ? errorState.message : ''

    const tabs = useMemo(() => {
        const supervised = evaluation?.supervised
        const classCount = supervised?.available ? supervised.original.per_class.length : null
        const objectCount = supervised?.object_analysis?.total_evaluable_objects
        return [
            { id: 'overview', label: 'Resumen' },
            { id: 'coco', label: 'Métricas COCO' },
            { id: 'classes', label: 'Por clase', badge: classCount },
            { id: 'curves', label: 'Curvas PR' },
            { id: 'objects', label: 'Por objeto', badge: objectCount },
            ...(supervised?.tide_analysis
                ? [{ id: 'tide', label: 'TIDE' }]
                : []),
            { id: 'images', label: 'Imágenes', badge: evaluation?.results?.length },
        ]
    }, [evaluation])

    async function handleEvaluate() {
        if (!selectedDataset) {
            setErrorState({ contextKey, message: 'Debes seleccionar un dataset.' })
            return
        }
        if (!isSupportedAttack) {
            setErrorState({
                contextKey,
                message:
                    'Para evaluar un detector selecciona Random Noise o YOLO Confidence.',
            })
            return
        }

        try {
            setLoading(true)
            setErrorState(null)
            eigenCamRun.current += 1
            eigenCamRequests.current.clear()
            dispatchEigenCam({ type: 'reset' })
            const result = await evaluateDetectionDataset({
                datasetId: selectedDataset.id,
                attackType: selectedAttack,
                epsilon: attackParameters.epsilon,
                alpha: attackParameters.alpha,
                steps: attackParameters.steps,
                supervisedConfiguration: configuration,
            })
            setEvaluationState({ contextKey, data: result })
            setActiveTab('overview')
        } catch (error) {
            setErrorState({ contextKey, message: error.message })
        } finally {
            setLoading(false)
        }
    }

    async function handleGenerateEigenCam(evaluationId, imageId) {
        const key = getDetectionEigenCamKey(evaluationId, imageId)
        const requestRun = eigenCamRun.current
        const currentEntry = getDetectionEigenCamEntry(eigenCamStates, key)

        if (
            !canRequestDetectionEigenCam(
                currentEntry,
                eigenCamRequests.current.has(key),
            )
        ) {
            return
        }

        eigenCamRequests.current.add(key)
        dispatchEigenCam({ type: 'loading', key })

        try {
            const response = await generateDetectionEvaluationEigenCam(
                evaluationId,
                imageId,
            )
            if (!isDetectionEigenCamResponse(response, evaluationId, imageId)) {
                throw new Error('Invalid EigenCAM response')
            }
            if (requestRun !== eigenCamRun.current) {
                return
            }
            dispatchEigenCam({ type: 'success', key, data: response })
        } catch (requestError) {
            if (requestRun !== eigenCamRun.current) {
                return
            }
            dispatchEigenCam({
                type: 'error',
                key,
                error: getDetectionEigenCamErrorMessage(requestError),
            })
        } finally {
            if (requestRun === eigenCamRun.current) {
                eigenCamRequests.current.delete(key)
            }
        }
    }

    function updateConfiguration(field, value) {
        setConfiguration((current) => ({ ...current, [field]: value }))
    }

    if (!selectedDataset) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Configuración incompleta</span>
                <h3>No hay dataset seleccionado</h3>
                <p>Vuelve al dashboard y selecciona un dataset antes de ejecutar la evaluación.</p>
            </div>
        )
    }

    return (
        <div className="evaluation-panel evaluation-dashboard detection-evaluation-dashboard">
            {!evaluation && !loading && (
                <section className="evaluation-run-card modern detection-run-card">
                    <div className="detection-run-copy">
                        <span className="evaluation-eyebrow">Evaluación de robustez COCO</span>
                        <h3>Analiza el detector con {attackName}</h3>
                        <p>
                            Se compararán predicciones originales y adversariales en{' '}
                            <strong>{selectedDataset.image_count}</strong> imágenes. Si las anotaciones son evaluables, se calcularán métricas COCO, diagnóstico TIDE y transiciones por objeto.
                        </p>
                        {!isSupportedAttack && (
                            <p className="muted-text">El ataque activo no es compatible con detección.</p>
                        )}
                    </div>

                    <div className="detection-run-controls" aria-label="Configuración supervisada">
                        <label>
                            <span>Confianza</span>
                            <input
                                type="number"
                                min="0.001"
                                max="1"
                                step="0.01"
                                value={configuration.confidence_threshold}
                                onChange={(event) =>
                                    updateConfiguration(
                                        'confidence_threshold',
                                        Number(event.target.value),
                                    )
                                }
                            />
                        </label>
                        <label>
                            <span>IoU</span>
                            <select
                                value={configuration.iou_threshold}
                                onChange={(event) =>
                                    updateConfiguration(
                                        'iou_threshold',
                                        Number(event.target.value),
                                    )
                                }
                            >
                                {IOU_THRESHOLDS.map((threshold) => (
                                    <option key={threshold} value={threshold}>
                                        {threshold.toFixed(2)}
                                    </option>
                                ))}
                            </select>
                        </label>
                        <label className="detection-details-toggle">
                            <input
                                type="checkbox"
                                checked={configuration.include_object_level_details}
                                onChange={(event) =>
                                    updateConfiguration(
                                        'include_object_level_details',
                                        event.target.checked,
                                    )
                                }
                            />
                            <span>Incluir detalle por objeto</span>
                        </label>
                        {!isConfigurationValid && (
                            <p className="detection-configuration-error" role="alert">
                                La confianza debe estar entre 0.001 y 1.
                            </p>
                        )}
                        <button
                            type="button"
                            className="primary-action evaluation-run-button"
                            onClick={handleEvaluate}
                            disabled={!isSupportedAttack || !isConfigurationValid}
                        >
                            Ejecutar evaluación
                        </button>
                    </div>
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
                        <strong>Evaluando detecciones y calculando COCO y TIDE</strong>
                        <p>El ataque y las inferencias se ejecutan una sola vez por imagen.</p>
                    </div>
                </section>
            )}

            {evaluation && (
                <section className="evaluation-results-shell">
                    <header className="detection-result-header">
                        <div>
                            <span className="evaluation-eyebrow">Evaluación completada</span>
                            <h3>{evaluation.model_name}</h3>
                            <p>ID {evaluation.result_id} · revisión {evaluation.dataset_revision}</p>
                        </div>
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={() => {
                                eigenCamRun.current += 1
                                eigenCamRequests.current.clear()
                                dispatchEigenCam({ type: 'reset' })
                                setEvaluationState(null)
                            }}
                        >
                            Nueva evaluación
                        </button>
                    </header>

                    <EvaluationTabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
                    <EvaluationMetricGuide
                        key={activeTab}
                        guide={DETECTION_METRIC_GUIDES[activeTab]}
                    />

                    <div
                        id={`evaluation-panel-${activeTab}`}
                        className="evaluation-tab-panel"
                        role="tabpanel"
                        aria-labelledby={`evaluation-tab-${activeTab}`}
                    >
                        {activeTab === 'overview' && <DetectionOverview evaluation={evaluation} />}
                        {activeTab === 'coco' && <DetectionCocoMetrics evaluation={evaluation} />}
                        {activeTab === 'classes' && <DetectionClassMetrics evaluation={evaluation} />}
                        {activeTab === 'curves' && <DetectionPrecisionRecall evaluation={evaluation} />}
                        {activeTab === 'objects' && <DetectionObjectAnalysis evaluation={evaluation} />}
                        {activeTab === 'tide' && <DetectionTideAnalysis evaluation={evaluation} />}
                        {activeTab === 'images' && (
                            <DetectionImageResults
                                evaluation={evaluation}
                                eigenCamStates={eigenCamStates}
                                onGenerateEigenCam={handleGenerateEigenCam}
                            />
                        )}
                    </div>
                </section>
            )}
        </div>
    )
}

export default DetectionEvaluationPanel
