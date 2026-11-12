import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import {
    evaluateDetectionDataset,
    generateDetectionEvaluationEigenCam,
} from '../api/detectionEvaluation'
import {
    findCompatibleEvaluation,
    getStoredEvaluation,
} from '../api/evaluationHistory'
import DetectionClassMetrics from './detectionEvaluation/DetectionClassMetrics'
import DetectionCocoMetrics from './detectionEvaluation/DetectionCocoMetrics'
import DetectionImageResults from './detectionEvaluation/DetectionImageResults'
import DetectionObjectAnalysis from './detectionEvaluation/DetectionObjectAnalysis'
import DetectionOverview from './detectionEvaluation/DetectionOverview'
import DetectionPrecisionRecall from './detectionEvaluation/DetectionPrecisionRecall'
import DetectionTideAnalysis from './detectionEvaluation/DetectionTideAnalysis'
import EvaluationInfoTip from './evaluation/EvaluationInfoTip'
import EvaluationMetricGuide from './evaluation/EvaluationMetricGuide'
import EvaluationProgress from './evaluation/EvaluationProgress'
import EvaluationTabs from './evaluation/EvaluationTabs'
import CompatibleEvaluationNotice from './evaluation/CompatibleEvaluationNotice'
import {
    canRequestDetectionEigenCam,
    detectionEigenCamReducer,
    getDetectionEigenCamEntry,
    getDetectionEigenCamErrorMessage,
    getDetectionEigenCamKey,
    isDetectionEigenCamResponse,
} from '../utils/detectionEvaluationEigenCamState'
import { getDetectionMetricGuide } from '../utils/evaluationMetricGuides'
import {
    attackParametersKey,
    isAttackSupportedForTask,
} from '../utils/attackCatalog'
import { getAttackName } from '../utils/attackFormatting'
import { isRequestCancelled } from '../api/client'
import { useNavigationGuard } from '../context/NavigationGuardContext'

const IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

function DetectionEvaluationPanel({
    selectedDataset,
    selectedAttack,
    modelId = null,
    attackParameters,
    attackCatalog = [],
    recoveredEvaluation = null,
    onStartNewEvaluation = () => {},
    onResultContextChange = () => {},
}) {
    const [evaluationState, setEvaluationState] = useState(null)
    const [activeTab, setActiveTab] = useState('overview')
    const [loading, setLoading] = useState(false)
    const { setNavigationProtection, clearNavigationProtection } =
        useNavigationGuard()
    const [progress, setProgress] = useState(null)
    const [errorState, setErrorState] = useState(null)
    const [reuseCandidate, setReuseCandidate] = useState(null)
    const [configuration, setConfiguration] = useState(
        recoveredEvaluation?.historyItem?.supervised_configuration ?? {
            confidence_threshold: 0.25,
            iou_threshold: 0.5,
            include_object_level_details: false,
        },
    )
    const [eigenCamStates, dispatchEigenCam] = useReducer(
        detectionEigenCamReducer,
        {},
    )
    const eigenCamRequests = useRef(new Set())
    const eigenCamRun = useRef(0)
    const evaluationAbortController = useRef(null)

    useEffect(() => {
        setNavigationProtection({ busy: loading })
    }, [loading, setNavigationProtection])

    useEffect(() => () => {
        evaluationAbortController.current?.abort()
        clearNavigationProtection()
    }, [clearNavigationProtection])
    const isSupportedAttack = isAttackSupportedForTask(
        attackCatalog,
        selectedAttack,
        'detection',
        modelId,
    )
    const isConfigurationValid =
        Number.isFinite(configuration.confidence_threshold) &&
        configuration.confidence_threshold >= 0.001 &&
        configuration.confidence_threshold <= 1
    const attackName = getAttackName(selectedAttack, attackCatalog)
    const contextKey = [
        selectedDataset?.id ?? '',
        selectedAttack,
        attackParametersKey(attackParameters),
        configuration.confidence_threshold,
        configuration.iou_threshold,
        configuration.include_object_level_details,
    ].join(':')

    const evaluation = recoveredEvaluation?.data ?? (
        evaluationState?.contextKey === contextKey ? evaluationState.data : null
    )
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

    async function runEvaluation(forceRecompute) {
        setReuseCandidate(null)
        setProgress(null)
        eigenCamRun.current += 1
        eigenCamRequests.current.clear()
        dispatchEigenCam({ type: 'reset' })
        const controller = new AbortController()
        evaluationAbortController.current = controller

        try {
            const result = await evaluateDetectionDataset({
                datasetId: selectedDataset.id,
                attackType: selectedAttack,
                parameters: attackParameters,
                supervisedConfiguration: configuration,
                forceRecompute,
                onProgress: setProgress,
                signal: controller.signal,
            })
            onStartNewEvaluation()
            setEvaluationState({ contextKey, data: result })
            onResultContextChange({
                resultId: result.result_id,
                datasetRevision: result.dataset_revision,
            })
            setActiveTab('overview')
        } finally {
            if (evaluationAbortController.current === controller) {
                evaluationAbortController.current = null
            }
        }
    }

    async function handleEvaluate() {
        if (!selectedDataset) {
            setErrorState({ contextKey, message: 'Debes seleccionar un conjunto de datos.' })
            return
        }
        if (!isSupportedAttack) {
            setErrorState({
                contextKey,
                message: 'El ataque seleccionado no es compatible con detección.',
            })
            return
        }

        try {
            setLoading(true)
            setErrorState(null)
            const compatibility = await findCompatibleEvaluation({
                taskType: 'detection',
                datasetId: selectedDataset.id,
                attackType: selectedAttack,
                parameters: attackParameters,
                supervisedConfiguration: configuration,
            })
            if (compatibility.match) {
                setReuseCandidate(compatibility.match)
                return
            }
            await runEvaluation(false)
        } catch (error) {
            if (!isRequestCancelled(error)) {
                setErrorState({ contextKey, message: error.message })
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
            if (stored.task_type !== 'detection') {
                throw new Error('El resultado compatible no es de detección.')
            }
            onStartNewEvaluation()
            setEvaluationState({ contextKey, data: stored.evaluation })
            onResultContextChange({
                resultId: stored.evaluation.result_id,
                datasetRevision: stored.evaluation.dataset_revision,
            })
            setReuseCandidate(null)
            setActiveTab('overview')
        } catch (requestError) {
            if (!isRequestCancelled(requestError)) {
                setErrorState({ contextKey, message: requestError.message })
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
        } catch (requestError) {
            if (!isRequestCancelled(requestError)) {
                setErrorState({ contextKey, message: requestError.message })
            }
        } finally {
            setLoading(false)
            setProgress(null)
        }
    }

    function handleCancelEvaluation() {
        evaluationAbortController.current?.abort()
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
                throw new Error('Respuesta de Eigen-CAM no válida')
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
                <h3>No hay conjunto de datos seleccionado</h3>
                <p>Vuelve al panel principal y selecciona un conjunto de datos antes de ejecutar la evaluación.</p>
            </div>
        )
    }

    return (
        <div className="evaluation-panel evaluation-dashboard detection-evaluation-dashboard detection-evaluation-panel">
            {!evaluation && !loading && (
                <section className="evaluation-run-card modern detection-run-card">
                    <div className="detection-run-copy">
                        <span className="evaluation-eyebrow">Evaluación de robustez COCO</span>
                        <h3>Analiza el detector con {attackName}</h3>
                        <p>
                            Se compararán predicciones originales y las obtenidas tras el ataque en{' '}
                            <strong>{selectedDataset.image_count}</strong> imágenes. Si las anotaciones son evaluables, se calcularán métricas COCO, diagnóstico TIDE y transiciones por objeto.
                        </p>
                        {!isSupportedAttack && (
                            <p className="muted-text">El ataque activo no es compatible con detección.</p>
                        )}
                    </div>

                    <div className="detection-run-controls" aria-label="Configuración supervisada">
                        <div className="detection-run-field">
                            <div className="detection-run-field-label">
                                <label htmlFor="detection-evaluation-confidence">Confianza</label>
                                <EvaluationInfoTip label="Confianza">
                                    Umbral mínimo de puntuación para incluir una detección en el punto de operación supervisado.
                                </EvaluationInfoTip>
                            </div>
                            <input
                                id="detection-evaluation-confidence"
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
                        </div>
                        <div className="detection-run-field">
                            <div className="detection-run-field-label">
                                <label htmlFor="detection-evaluation-iou">IoU de matching</label>
                                <EvaluationInfoTip label="IoU de matching">
                                    Solapamiento mínimo entre una detección y una caja de referencia para considerarlas emparejadas.
                                </EvaluationInfoTip>
                            </div>
                            <select
                                id="detection-evaluation-iou"
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
                        </div>
                        <div className="detection-details-toggle-row">
                            <label className="detection-details-toggle" htmlFor="detection-evaluation-object-details">
                                <input
                                    id="detection-evaluation-object-details"
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
                            <EvaluationInfoTip label="Detalle por objeto">
                                Materializa las tablas por imagen y el explorador de objetos. Aumenta el detalle de la respuesta, no cambia las métricas COCO.
                            </EvaluationInfoTip>
                        </div>
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
                    title="Evaluando detecciones y calculando COCO y TIDE"
                    pendingMessage="Preparando la evaluación del conjunto de datos..."
                    onCancel={handleCancelEvaluation}
                />
            )}

            {evaluation && !loading && (
                <section className="evaluation-results-shell" aria-label="Resultados de evaluación de detección">
                    <p className="evaluation-sr-only" role="status" aria-live="polite">
                        Evaluación de detección completada. Se muestran los resultados del conjunto de datos.
                    </p>

                    <EvaluationTabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
                    <EvaluationMetricGuide
                        key={activeTab}
                        guide={getDetectionMetricGuide(activeTab, evaluation)}
                        variant="detection"
                    />

                    <div
                        id={`evaluation-panel-${activeTab}`}
                        className="evaluation-tab-panel"
                        role="tabpanel"
                        aria-labelledby={`evaluation-tab-${activeTab}`}
                        tabIndex={0}
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

                    <footer className="evaluation-rerun-footer detection-evaluation-rerun-footer">
                        <div>
                            <strong>Resultado disponible</strong>
                            <span>
                                Puedes repetir la evaluación con la misma configuración si necesitas resultados nuevos.
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

export default DetectionEvaluationPanel
