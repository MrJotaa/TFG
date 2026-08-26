import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { applyAttack } from '../api/attack'
import { isRequestCancelled, resolveApiUrl } from '../api/client'
import { attackDetectionImage, detectImage } from '../api/detect'
import { generateDetectionExplainability, generateGradCam } from '../api/explainability'
import { predictImage } from '../api/predict'
import AdversarialImageExplorer from '../components/visualExplorer/AdversarialImageExplorer'
import { useExperiment } from '../context/ExperimentContext'
import { useNavigationGuard } from '../context/NavigationGuardContext'
import { resolveCleanDetectionResult } from '../utils/adversarialExplorer'
import {
    attackParametersKey,
    isAttackSupportedForTask,
} from '../utils/attackCatalog'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import {
    analysisImagePosition,
    formatAnalysisProbability,
    formatSignedInteger,
    formatSignedPercentagePoints,
    normalizeClassificationPanel,
    normalizeDetectionPanel,
    resolveClassificationExplanation,
    resolveDetectionExplanation,
} from '../utils/individualAnalysis'

function DetectionList({ detections = [], title }) {
    return (
        <section className="detection-result-list">
            <header>
                <h4>{title}</h4>
                <span>{detections.length}</span>
            </header>
            <div className="detection-result-rows">
                {detections.length ? detections.map((detection, index) => (
                    <div
                        key={`${detection.label}-${index}`}
                        className="detection-result-row"
                    >
                        <div>
                            <strong>{detection.label}</strong>
                            <small>
                                {detection.bbox.x1}, {detection.bbox.y1} →{' '}
                                {detection.bbox.x2}, {detection.bbox.y2}
                            </small>
                        </div>
                        <span>{formatAnalysisProbability(detection.confidence)}</span>
                    </div>
                )) : (
                    <p className="detection-result-empty">No hay detecciones en esta condición.</p>
                )}
            </div>
        </section>
    )
}

function ClassificationTopK({ title, predictions = [] }) {
    return (
        <section className="classification-topk-column">
            <h4>{title}</h4>
            <div className="classification-topk-list">
                {predictions.map((prediction) => {
                    const probability = Number(prediction.probability)
                    const width = Number.isFinite(probability)
                        ? `${Math.max(0, Math.min(100, probability * 100))}%`
                        : '0%'

                    return (
                        <div key={prediction.label} className="classification-topk-row">
                            <div>
                                <span>{prediction.label}</span>
                                <strong>{formatAnalysisProbability(prediction.probability)}</strong>
                            </div>
                            <span className="classification-topk-track" aria-hidden="true">
                                <span style={{ width }} />
                            </span>
                        </div>
                    )
                })}
            </div>
        </section>
    )
}

function ClassificationResults({ result }) {
    const comparison = result.comparison

    return (
        <section className="classification-impact-section" aria-labelledby="classification-impact-title">
            <header className="classification-section-heading">
                <div>
                    <span className="evaluation-eyebrow">Resultado</span>
                    <h3 id="classification-impact-title">Impacto en la predicción</h3>
                </div>
                <span
                    className={`classification-attack-status ${comparison.attack_success ? 'danger' : 'success'}`}
                >
                    <span aria-hidden="true">●</span>
                    {comparison.attack_success ? 'Ataque exitoso' : 'Predicción preservada'}
                </span>
            </header>

            <div className="classification-impact-flow">
                <div className="classification-impact-state">
                    <span>Original</span>
                    <strong>{result.original_prediction.prediction}</strong>
                    <small>{formatAnalysisProbability(comparison.original_confidence)}</small>
                </div>
                <div className="classification-impact-delta" aria-label="Cambio de confianza">
                    <span aria-hidden="true">→</span>
                    <strong>{formatSignedPercentagePoints(comparison.confidence_difference)}</strong>
                    <small>Cambio de confianza</small>
                </div>
                <div className={`classification-impact-state ${comparison.prediction_changed ? 'changed' : ''}`}>
                    <span>Tras el ataque</span>
                    <strong>{result.adversarial_prediction.prediction}</strong>
                    <small>{formatAnalysisProbability(comparison.adversarial_confidence)}</small>
                </div>
            </div>

            <div className="classification-topk-comparison">
                <ClassificationTopK
                    title="Top-K original"
                    predictions={result.original_prediction.top_k}
                />
                <ClassificationTopK
                    title="Top-K tras el ataque"
                    predictions={result.adversarial_prediction.top_k}
                />
            </div>
        </section>
    )
}

function CleanPredictionResult({ prediction, selectedImage }) {
    return (
        <section className="classification-prediction-result" aria-live="polite">
            <figure className="classification-prediction-figure">
                <img
                    src={resolveApiUrl(selectedImage.url)}
                    alt={selectedImage.filename}
                />
                <figcaption>{selectedImage.filename}</figcaption>
            </figure>

            <div className="classification-prediction-summary">
                <span className="evaluation-eyebrow">Predicción original</span>
                <h3>{prediction.prediction}</h3>
                <p className="classification-prediction-confidence">
                    {formatAnalysisProbability(prediction.confidence)}
                    <span>de confianza</span>
                </p>
                <ClassificationTopK
                    title="Top predicciones"
                    predictions={prediction.top_k}
                />
            </div>
        </section>
    )
}

function ClassificationExplainabilityResult({ explanation, selectedImage }) {
    if (!explanation) {
        return null
    }

    const items = explanation.mode === 'comparison'
        ? [
            ['Original', explanation.original],
            ['Tras el ataque', explanation.adversarial],
        ]
        : [['Grad-CAM original', explanation.original]]

    return (
        <section className={`classification-explainability-result ${explanation.mode}`}>
            <div className="classification-explanation-grid">
                {items.map(([title, item]) => (
                    <figure key={title} className="classification-explanation-figure">
                        <figcaption>
                            <span>{title}</span>
                            {item?.prediction && (
                                <small>
                                    {item.prediction}
                                    {item.confidence != null
                                        ? ` · ${formatAnalysisProbability(item.confidence)}`
                                        : ''}
                                </small>
                            )}
                        </figcaption>
                        <img
                            src={resolveApiUrl(item?.imageUrl)}
                            alt={`${title} de ${selectedImage.filename}`}
                        />
                    </figure>
                ))}
            </div>
            <p className="classification-explanation-note">
                {explanation.mode === 'comparison'
                    ? 'Compara la atención del modelo antes y después del ataque. La comparación visual también sigue disponible dentro del explorador adversario.'
                    : 'Grad-CAM resalta las regiones que más influyen en la predicción de la imagen original.'}
            </p>
        </section>
    )
}

function ClassificationWorkspaceTabs({ activePanel, onSelect }) {
    const tabs = [
        ['clean', 'Predicción'],
        ['attack', 'Comparación'],
        ['explanation', 'Explicabilidad'],
    ]

    return (
        <nav className="classification-analysis-tabs" aria-label="Modos del análisis individual">
            {tabs.map(([panel, label]) => (
                <button
                    key={panel}
                    type="button"
                    className={activePanel === panel ? 'active' : ''}
                    aria-current={activePanel === panel ? 'page' : undefined}
                    onClick={() => onSelect(panel)}
                >
                    {label}
                </button>
            ))}
        </nav>
    )
}

function ClassificationLoadingState({ title, description, onCancel }) {
    return (
        <div className="classification-inline-loading" role="status">
            <span className="evaluation-loading-spinner" aria-hidden="true" />
            <div>
                <strong>{title}</strong>
                <p>{description}</p>
            </div>
            <button type="button" className="secondary-action" onClick={onCancel}>
                Cancelar
            </button>
        </div>
    )
}

function AnalysisPrompt({ eyebrow, title, description, actionLabel, onAction, disabled = false }) {
    return (
        <div className="classification-analysis-prompt">
            <span className="evaluation-eyebrow">{eyebrow}</span>
            <h3>{title}</h3>
            <p>{description}</p>
            <button
                type="button"
                className="primary-action"
                onClick={onAction}
                disabled={disabled}
            >
                {actionLabel}
            </button>
        </div>
    )
}

function CleanDetectionResult({ detection, selectedImage }) {
    const detections = detection.detections ?? []

    return (
        <section className="detection-clean-result" aria-live="polite">
            <figure className="detection-clean-figure">
                <img
                    src={resolveApiUrl(
                        detection.annotated_image_url ?? selectedImage.url,
                    )}
                    alt={`Detecciones sobre ${selectedImage.filename}`}
                />
                <figcaption>{selectedImage.filename}</figcaption>
            </figure>

            <div className="detection-clean-summary">
                <span className="evaluation-eyebrow">Detección original</span>
                <h3>
                    {detections.length}
                    <span>{detections.length === 1 ? 'objeto detectado' : 'objetos detectados'}</span>
                </h3>
                <p>
                    Las cajas corresponden a la inferencia original y conservan la
                    geometría de la imagen seleccionada.
                </p>
                <DetectionList
                    title="Detecciones"
                    detections={detections}
                />
            </div>
        </section>
    )
}

function DetectionExplainabilityResult({ explanation, selectedImage }) {
    if (!explanation) return null

    const items = explanation.mode === 'comparison'
        ? [
            ['Original', explanation.original],
            ['Tras el ataque', explanation.adversarial],
        ]
        : [['Eigen-CAM original', explanation.original]]

    return (
        <section className={`detection-explainability-result ${explanation.mode}`} aria-live="polite">
            <div className="detection-explanation-grid">
                {items.map(([title, item]) => (
                    <figure key={title} className="detection-explanation-figure">
                        <figcaption>
                            <span>{title}</span>
                            <small>{item?.detections?.length ?? 0} detecciones</small>
                        </figcaption>
                        <img
                            src={resolveApiUrl(item?.imageUrl)}
                            alt={`${title} de ${selectedImage.filename}`}
                        />
                    </figure>
                ))}
            </div>

            <div className={`detection-explanation-lists ${explanation.mode}`}>
                {items.map(([title, item]) => (
                    <DetectionList
                        key={`${title}-detections`}
                        title={title === 'Original' ? 'Detecciones originales' : title === 'Tras el ataque' ? 'Detecciones tras el ataque' : 'Detecciones explicadas'}
                        detections={item?.detections ?? []}
                    />
                ))}
            </div>

            <p className="detection-explanation-note">
                {explanation.mode === 'comparison'
                    ? 'Compara las regiones de mayor influencia antes y después del ataque. Eigen-CAM también permanece disponible dentro del visor adversario.'
                    : 'Eigen-CAM resalta las regiones con mayor influencia sobre las detecciones de la imagen original.'}
            </p>
        </section>
    )
}

function DetectionWorkspaceTabs({ activePanel, onSelect }) {
    const tabs = [
        ['clean', 'Detección'],
        ['attack', 'Comparación'],
        ['explanation', 'Explicabilidad'],
    ]

    return (
        <nav className="classification-analysis-tabs" aria-label="Modos del análisis individual de detección">
            {tabs.map(([panel, label]) => (
                <button
                    key={panel}
                    type="button"
                    className={activePanel === panel ? 'active' : ''}
                    aria-current={activePanel === panel ? 'page' : undefined}
                    onClick={() => onSelect(panel)}
                >
                    {label}
                </button>
            ))}
        </nav>
    )
}

function DetectionResults({ result }) {
    const comparison = result.comparison
    const countDelta = comparison.adversarial_detection_count - comparison.original_detection_count
    const confidenceDelta = comparison.average_adversarial_confidence - comparison.average_original_confidence

    return (
        <section className="detection-impact-section" aria-labelledby="detection-impact-title">
            <header className="classification-section-heading">
                <div>
                    <span className="evaluation-eyebrow">Resultado</span>
                    <h3 id="detection-impact-title">Impacto en las detecciones</h3>
                </div>
                <span
                    className={`classification-attack-status ${comparison.attack_success ? 'danger' : 'success'}`}
                >
                    <span aria-hidden="true">●</span>
                    {comparison.attack_success ? 'Ataque exitoso' : 'Detección preservada'}
                </span>
            </header>

            <div className="detection-impact-flow">
                <div className="detection-impact-state">
                    <span>Original</span>
                    <strong>{comparison.original_detection_count}</strong>
                    <small>detecciones</small>
                </div>
                <div className="detection-impact-delta">
                    <span aria-hidden="true">→</span>
                    <strong>{formatSignedInteger(countDelta)}</strong>
                    <small>cambio de detecciones</small>
                </div>
                <div className={`detection-impact-state ${comparison.attack_success ? 'changed' : ''}`}>
                    <span>Tras el ataque</span>
                    <strong>{comparison.adversarial_detection_count}</strong>
                    <small>detecciones</small>
                </div>
            </div>

            <div className="detection-confidence-flow">
                <span>Confianza media</span>
                <strong>{formatAnalysisProbability(comparison.average_original_confidence)}</strong>
                <span aria-hidden="true">→</span>
                <strong>{formatAnalysisProbability(comparison.average_adversarial_confidence)}</strong>
                <em>{formatSignedPercentagePoints(confidenceDelta)}</em>
            </div>

            <p className="detection-impact-note">
                El recuento y la confianza son descriptivos para esta imagen. Sin
                matching contra anotaciones de referencia no equivalen a AP ni recall.
            </p>

            <div className="detection-result-comparison">
                <DetectionList
                    title="Detecciones originales"
                    detections={result.original_detections}
                />
                <DetectionList
                    title="Detecciones tras el ataque"
                    detections={result.adversarial_detections}
                />
            </div>
        </section>
    )
}

function ComparisonView() {
    const navigate = useNavigate()
    const {
        selectedDataset,
        selectedImage,
        setSelectedImage,
        activeModel,
        attackCatalog,
        selectedAttack,
        attackParameters,
    } = useExperiment()
    const { setNavigationProtection, clearNavigationProtection, requestNavigation } =
        useNavigationGuard()
    const [predictionResultState, setPredictionResultState] = useState(null)
    const [attackResultState, setAttackResultState] = useState(null)
    const [explainabilityResultState, setExplainabilityResultState] = useState(null)
    const [activePanelState, setActivePanelState] = useState(null)
    const [contextOpen, setContextOpen] = useState(false)
    const [loadingPrediction, setLoadingPrediction] = useState(false)
    const [loading, setLoading] = useState(false)
    const [loadingExplainability, setLoadingExplainability] = useState(false)
    const [errorState, setErrorState] = useState(null)
    const requestAbortController = useRef(null)

    const taskType = activeModel?.task_type ?? 'classification'
    const isDetection = taskType === 'detection'
    const isAttackSupported = isAttackSupportedForTask(
        attackCatalog,
        selectedAttack,
        taskType,
    )
    const datasetImages = useMemo(
        () => selectedDataset?.images ?? [],
        [selectedDataset],
    )
    const selectedIndex = datasetImages.findIndex(
        (image) => image.id === selectedImage?.id,
    )
    const imageContextKey = [
        selectedDataset?.id,
        selectedImage?.id,
        activeModel?.id,
    ].join(':')
    const attackContextKey = [
        imageContextKey,
        selectedAttack,
        attackParametersKey(attackParameters),
    ].join(':')
    const predictionResult = predictionResultState?.contextKey === imageContextKey
        ? predictionResultState.data
        : null
    const attackResult = attackResultState?.contextKey === attackContextKey
        ? attackResultState.data
        : null
    const explainabilityResult =
        explainabilityResultState?.contextKey === imageContextKey
            ? explainabilityResultState.data
            : null
    const cleanPrediction = !isDetection
        ? attackResult?.original_prediction ?? predictionResult
        : null
    const cleanDetection = isDetection
        ? resolveCleanDetectionResult({
            detectionResult: predictionResult,
            attackResult,
        })
        : null
    const requestedPanel = activePanelState?.contextKey === imageContextKey
        ? activePanelState.panel
        : 'clean'
    const activePanel = !isDetection
        ? normalizeClassificationPanel(requestedPanel)
        : normalizeDetectionPanel(requestedPanel)
    const classificationExplanation = !isDetection
        ? resolveClassificationExplanation({
            attackResult,
            explainabilityResult,
        })
        : null
    const detectionExplanation = isDetection
        ? resolveDetectionExplanation({
            attackResult,
            explainabilityResult,
        })
        : null
    const error = errorState?.contextKey === imageContextKey
        ? errorState.message
        : ''
    const busy = loading || loadingPrediction || loadingExplainability
    const hasTemporaryWork = Boolean(
        predictionResult || attackResult || explainabilityResult,
    )

    useEffect(() => {
        setNavigationProtection({ hasTemporaryWork, busy })
    }, [hasTemporaryWork, busy, setNavigationProtection])

    useEffect(() => () => {
        requestAbortController.current?.abort()
        clearNavigationProtection()
    }, [clearNavigationProtection])

    function beginCancellableRequest() {
        requestAbortController.current?.abort()
        const controller = new AbortController()
        requestAbortController.current = controller
        return controller
    }

    function finishCancellableRequest(controller) {
        if (requestAbortController.current === controller) {
            requestAbortController.current = null
        }
    }

    function handleCancelOperation() {
        requestAbortController.current?.abort()
    }

    function setCurrentError(message) {
        setErrorState(message ? { contextKey: imageContextKey, message } : null)
    }

    async function handlePredict() {
        if (!selectedDataset || !selectedImage || !activeModel) {
            setCurrentError('Debes seleccionar un conjunto de datos, una imagen y un modelo.')
            return
        }

        const controller = beginCancellableRequest()
        try {
            setLoadingPrediction(true)
            setCurrentError('')
            setPredictionResultState(null)
            if (!isDetection) {
                setActivePanelState({ contextKey: imageContextKey, panel: 'clean' })
            }
            const request = {
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
                signal: controller.signal,
            }
            const result = isDetection
                ? await detectImage(request)
                : await predictImage(request)
            setPredictionResultState({ contextKey: imageContextKey, data: result })
            setActivePanelState({ contextKey: imageContextKey, panel: 'clean' })
        } catch (err) {
            if (!isRequestCancelled(err)) {
                setCurrentError(err.message)
            }
        } finally {
            finishCancellableRequest(controller)
            setLoadingPrediction(false)
        }
    }

    async function handleRunAttack() {
        if (!selectedDataset || !selectedImage || !activeModel) {
            setCurrentError('Debes seleccionar un conjunto de datos, una imagen y un modelo.')
            return
        }

        if (!isAttackSupported) {
            setCurrentError('El ataque seleccionado no es compatible con el modelo activo.')
            return
        }

        const controller = beginCancellableRequest()
        const request = {
            datasetId: selectedDataset.id,
            imageId: selectedImage.id,
            attackType: selectedAttack,
            parameters: attackParameters,
            signal: controller.signal,
        }

        try {
            setLoading(true)
            setCurrentError('')
            setAttackResultState(null)
            setActivePanelState({ contextKey: imageContextKey, panel: 'attack' })
            const result = isDetection
                ? await attackDetectionImage(request)
                : await applyAttack(request)
            setAttackResultState({ contextKey: attackContextKey, data: result })
        } catch (err) {
            if (!isRequestCancelled(err)) {
                setCurrentError(err.message)
            }
        } finally {
            finishCancellableRequest(controller)
            setLoading(false)
        }
    }

    async function handleGenerateExplainability() {
        if (!selectedDataset || !selectedImage || !activeModel) {
            setCurrentError('Debes seleccionar un conjunto de datos, una imagen y un modelo.')
            return
        }

        const controller = beginCancellableRequest()
        try {
            setLoadingExplainability(true)
            setCurrentError('')
            setActivePanelState({
                contextKey: imageContextKey,
                panel: 'explanation',
            })
            const request = {
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
                signal: controller.signal,
            }
            const result = isDetection
                ? await generateDetectionExplainability(request)
                : await generateGradCam(request)
            setExplainabilityResultState({
                contextKey: imageContextKey,
                data: result,
            })
        } catch (err) {
            if (!isRequestCancelled(err)) {
                setCurrentError(err.message)
            }
        } finally {
            finishCancellableRequest(controller)
            setLoadingExplainability(false)
        }
    }

    function selectRelativeImage(offset) {
        const nextImage = datasetImages[selectedIndex + offset]

        if (nextImage) {
            setSelectedImage(nextImage)
        }
    }

    function navigateToDashboard() {
        requestNavigation(() => navigate('/dashboard'))
    }

    const imageDimensions = selectedImage?.width && selectedImage?.height
        ? `${selectedImage.width} × ${selectedImage.height} px`
        : null
    const attackName = getAttackName(selectedAttack, attackCatalog)
    const attackParameterSummary = formatAttackParameters(
        selectedAttack,
        attackParameters,
        attackCatalog,
    )

    if (!isDetection) {
        const hasContext = Boolean(selectedDataset && selectedImage && activeModel)

        return (
            <section className="comparison-page classification-analysis-page">
                <header className="classification-analysis-header">
                    <span className="evaluation-eyebrow">Análisis individual · Clasificación</span>
                    <h2>Predicción, robustez y atención del modelo</h2>
                    <p>
                        Explora la inferencia original, compara el efecto del ataque y
                        analiza Grad-CAM sin abandonar la misma imagen.
                    </p>
                </header>

                {!hasContext ? (
                    <div className="classification-analysis-missing">
                        <span className="evaluation-eyebrow">Contexto incompleto</span>
                        <h3>Selecciona el experimento antes de analizar</h3>
                        <p>
                            Necesitas un conjunto de datos, una imagen y un modelo activo.
                        </p>
                        <button
                            type="button"
                            className="primary-action"
                            onClick={navigateToDashboard}
                        >
                            Configurar experimento
                        </button>
                    </div>
                ) : (
                    <>
                        <section className="classification-context-shell">
                            <div className="classification-context-strip">
                                <div className="classification-context-summary">
                                    <strong title={activeModel.name}>{activeModel.name}</strong>
                                    <span aria-hidden="true">/</span>
                                    <span>{attackName} {attackParameterSummary}</span>
                                    <span aria-hidden="true">/</span>
                                    <span>{selectedDataset.name}</span>
                                </div>

                                <div className="classification-context-actions">
                                    <div className="classification-image-nav" aria-label="Navegación entre imágenes">
                                        <button
                                            type="button"
                                            onClick={() => selectRelativeImage(-1)}
                                            disabled={selectedIndex <= 0 || busy}
                                            aria-label="Imagen anterior"
                                            title="Imagen anterior"
                                        >
                                            ‹
                                        </button>
                                        <span title={selectedImage.filename}>
                                            {analysisImagePosition(selectedIndex, datasetImages.length)}
                                        </span>
                                        <button
                                            type="button"
                                            onClick={() => selectRelativeImage(1)}
                                            disabled={
                                                selectedIndex < 0 ||
                                                selectedIndex >= datasetImages.length - 1 ||
                                                busy
                                            }
                                            aria-label="Imagen siguiente"
                                            title="Imagen siguiente"
                                        >
                                            ›
                                        </button>
                                    </div>
                                    <button
                                        type="button"
                                        className="classification-context-toggle"
                                        aria-expanded={contextOpen}
                                        onClick={() => setContextOpen((current) => !current)}
                                    >
                                        Contexto
                                        <span aria-hidden="true">{contextOpen ? '▴' : '▾'}</span>
                                    </button>
                                </div>
                            </div>

                            {contextOpen && (
                                <dl className="classification-context-details">
                                    <div>
                                        <dt>Imagen</dt>
                                        <dd>{selectedImage.filename}</dd>
                                    </div>
                                    <div>
                                        <dt>Geometría</dt>
                                        <dd>{imageDimensions || 'No disponible'}</dd>
                                    </div>
                                    <div>
                                        <dt>Modelo</dt>
                                        <dd>{activeModel.name}</dd>
                                    </div>
                                    <div>
                                        <dt>Ataque</dt>
                                        <dd>{attackName}</dd>
                                    </div>
                                    <div>
                                        <dt>Parámetros</dt>
                                        <dd>{attackParameterSummary || 'Configuración por defecto'}</dd>
                                    </div>
                                    <div>
                                        <dt>Dataset</dt>
                                        <dd>{selectedDataset.name}</dd>
                                    </div>
                                    {attackResult?.result_id && (
                                        <div>
                                            <dt>Resultado temporal</dt>
                                            <dd>{attackResult.result_id}</dd>
                                        </div>
                                    )}
                                </dl>
                            )}
                        </section>

                        {error && <p className="error classification-analysis-error" role="alert">{error}</p>}

                        {!isAttackSupported && (
                            <div className="evaluation-warning-banner">
                                El ataque seleccionado no es compatible con el modelo activo.
                                Cámbialo desde el panel principal antes de ejecutar la comparación.
                            </div>
                        )}

                        <ClassificationWorkspaceTabs
                            activePanel={activePanel}
                            onSelect={(panel) => setActivePanelState({
                                contextKey: imageContextKey,
                                panel,
                            })}
                        />

                        <div className="classification-analysis-workspace">
                            {activePanel === 'clean' && (
                                <section className="classification-analysis-mode">
                                    <header className="classification-mode-heading">
                                        <div>
                                            <span className="evaluation-eyebrow">Predicción</span>
                                            <h3>Qué reconoce el modelo</h3>
                                            <p>
                                                Ejecuta únicamente la inferencia original cuando no necesites aplicar un ataque.
                                            </p>
                                        </div>
                                        {cleanPrediction && !loadingPrediction && (
                                            <button
                                                type="button"
                                                className="secondary-action"
                                                onClick={handlePredict}
                                                disabled={busy}
                                            >
                                                Repetir predicción
                                            </button>
                                        )}
                                    </header>

                                    {loadingPrediction ? (
                                        <ClassificationLoadingState
                                            title="Calculando predicción original"
                                            description="El resultado aparecerá en este mismo espacio de análisis."
                                            onCancel={handleCancelOperation}
                                        />
                                    ) : cleanPrediction ? (
                                        <CleanPredictionResult
                                            prediction={cleanPrediction}
                                            selectedImage={selectedImage}
                                        />
                                    ) : (
                                        <AnalysisPrompt
                                            eyebrow="Inferencia original"
                                            title="Predice esta imagen"
                                            description="Obtén la clase predicha, su confianza y el Top-K sin modificar la imagen."
                                            actionLabel="Ejecutar predicción"
                                            onAction={handlePredict}
                                        />
                                    )}
                                </section>
                            )}

                            {activePanel === 'attack' && (
                                <section className="classification-analysis-mode">
                                    <header className="classification-mode-heading">
                                        <div>
                                            <span className="evaluation-eyebrow">Comparación adversaria</span>
                                            <h3>Original frente al resultado tras el ataque</h3>
                                            <p>
                                                Conserva el visor sincronizado, la superposición y Grad-CAM para comparar ambas condiciones.
                                            </p>
                                        </div>
                                        {attackResult && !loading && (
                                            <button
                                                type="button"
                                                className="secondary-action"
                                                onClick={handleRunAttack}
                                                disabled={busy || !isAttackSupported}
                                            >
                                                Repetir ataque
                                            </button>
                                        )}
                                    </header>

                                    {loading ? (
                                        <ClassificationLoadingState
                                            title={`Aplicando ${attackName}`}
                                            description="Se calculan la referencia original, la imagen adversaria, Grad-CAM y la perturbación."
                                            onCancel={handleCancelOperation}
                                        />
                                    ) : attackResult ? (
                                        <>
                                            <AdversarialImageExplorer
                                                key={attackResult.result_id}
                                                taskType={taskType}
                                                selectedImage={selectedImage}
                                                attackResult={attackResult}
                                                filename={selectedImage.filename}
                                                imageDimensions={imageDimensions}
                                                perturbationMetrics={{
                                                    linf: attackResult.comparison.perturbation_linf,
                                                    l2: attackResult.comparison.perturbation_l2,
                                                }}
                                            />
                                            <ClassificationResults result={attackResult} />
                                        </>
                                    ) : (
                                        <AnalysisPrompt
                                            eyebrow="Robustez adversaria"
                                            title="Compara la imagen frente al ataque"
                                            description={`El ataque ${attackName} calculará también la predicción original y los Grad-CAM necesarios para la exploración visual.`}
                                            actionLabel={`Aplicar ${attackName} y comparar`}
                                            onAction={handleRunAttack}
                                            disabled={!isAttackSupported}
                                        />
                                    )}
                                </section>
                            )}

                            {activePanel === 'explanation' && (
                                <section className="classification-analysis-mode">
                                    <header className="classification-mode-heading">
                                        <div>
                                            <span className="evaluation-eyebrow">Explicabilidad</span>
                                            <h3>Dónde se fija el modelo</h3>
                                            <p>
                                                Analiza Grad-CAM como resultado independiente o compara la atención original y adversaria cuando ya existe un ataque.
                                            </p>
                                        </div>
                                        {classificationExplanation?.mode === 'single' &&
                                            !loadingExplainability && (
                                            <button
                                                type="button"
                                                className="secondary-action"
                                                onClick={handleGenerateExplainability}
                                                disabled={busy}
                                            >
                                                Regenerar Grad-CAM
                                            </button>
                                        )}
                                    </header>

                                    {loadingExplainability ? (
                                        <ClassificationLoadingState
                                            title="Generando Grad-CAM"
                                            description="El mapa resaltará las regiones con mayor influencia sobre la predicción original."
                                            onCancel={handleCancelOperation}
                                        />
                                    ) : classificationExplanation ? (
                                        <ClassificationExplainabilityResult
                                            explanation={classificationExplanation}
                                            selectedImage={selectedImage}
                                        />
                                    ) : (
                                        <AnalysisPrompt
                                            eyebrow="Atención visual"
                                            title="Genera Grad-CAM para esta imagen"
                                            description="Obtén una explicación visual de la predicción sin necesidad de ejecutar un ataque."
                                            actionLabel="Generar Grad-CAM"
                                            onAction={handleGenerateExplainability}
                                        />
                                    )}
                                </section>
                            )}
                        </div>
                    </>
                )}
            </section>
        )
    }

    const hasDetectionContext = Boolean(selectedDataset && selectedImage && activeModel)

    return (
        <section className="comparison-page classification-analysis-page detection-analysis-page">
            <header className="classification-analysis-header">
                <span className="evaluation-eyebrow">Análisis individual · Detección</span>
                <h2>Detección, robustez y atención del modelo</h2>
                <p>
                    Explora las detecciones originales, compara el efecto del ataque y
                    analiza Eigen-CAM sin abandonar la misma imagen.
                </p>
            </header>

            {!hasDetectionContext ? (
                <div className="classification-analysis-missing">
                    <span className="evaluation-eyebrow">Contexto incompleto</span>
                    <h3>Selecciona el experimento antes de analizar</h3>
                    <p>
                        Necesitas un conjunto de datos, una imagen y un detector activo.
                    </p>
                    <button
                        type="button"
                        className="primary-action"
                        onClick={navigateToDashboard}
                    >
                        Configurar experimento
                    </button>
                </div>
            ) : (
                <>
                    <section className="classification-context-shell">
                        <div className="classification-context-strip">
                            <div className="classification-context-summary">
                                <strong title={activeModel.name}>{activeModel.name}</strong>
                                <span aria-hidden="true">/</span>
                                <span>{attackName} {attackParameterSummary}</span>
                                <span aria-hidden="true">/</span>
                                <span>{selectedDataset.name}</span>
                            </div>

                            <div className="classification-context-actions">
                                <div className="classification-image-nav" aria-label="Navegación entre imágenes">
                                    <button
                                        type="button"
                                        onClick={() => selectRelativeImage(-1)}
                                        disabled={selectedIndex <= 0 || busy}
                                        aria-label="Imagen anterior"
                                        title="Imagen anterior"
                                    >
                                        ‹
                                    </button>
                                    <span title={selectedImage.filename}>
                                        {analysisImagePosition(selectedIndex, datasetImages.length)}
                                    </span>
                                    <button
                                        type="button"
                                        onClick={() => selectRelativeImage(1)}
                                        disabled={
                                            selectedIndex < 0 ||
                                            selectedIndex >= datasetImages.length - 1 ||
                                            busy
                                        }
                                        aria-label="Imagen siguiente"
                                        title="Imagen siguiente"
                                    >
                                        ›
                                    </button>
                                </div>

                                <button
                                    type="button"
                                    className="classification-context-toggle"
                                    aria-expanded={contextOpen}
                                    onClick={() => setContextOpen((current) => !current)}
                                >
                                    Contexto
                                    <span aria-hidden="true">{contextOpen ? '▴' : '▾'}</span>
                                </button>
                            </div>
                        </div>

                        {contextOpen && (
                            <dl className="classification-context-details">
                                <div>
                                    <dt>Imagen</dt>
                                    <dd>{selectedImage.filename}</dd>
                                </div>
                                <div>
                                    <dt>Geometría</dt>
                                    <dd>{imageDimensions || 'No disponible'}</dd>
                                </div>
                                <div>
                                    <dt>Modelo</dt>
                                    <dd>{activeModel.name}</dd>
                                </div>
                                <div>
                                    <dt>Ataque</dt>
                                    <dd>{attackName}</dd>
                                </div>
                                <div>
                                    <dt>Parámetros</dt>
                                    <dd>{attackParameterSummary || 'Sin parámetros'}</dd>
                                </div>
                                <div>
                                    <dt>Dataset</dt>
                                    <dd>{selectedDataset.name}</dd>
                                </div>
                                {(attackResult?.result_id || explainabilityResult?.result_id) && (
                                    <div>
                                        <dt>Resultado temporal</dt>
                                        <dd>{attackResult?.result_id || explainabilityResult?.result_id}</dd>
                                    </div>
                                )}
                            </dl>
                        )}
                    </section>

                    {error && (
                        <p className="error classification-analysis-error" role="alert">
                            {error}
                        </p>
                    )}

                    {!isAttackSupported && (
                        <div className="evaluation-warning-banner">
                            El ataque seleccionado no es compatible con este detector.
                            Cambia el ataque desde el panel principal.
                        </div>
                    )}

                    <DetectionWorkspaceTabs
                        activePanel={activePanel}
                        onSelect={(panel) => setActivePanelState({
                            contextKey: imageContextKey,
                            panel,
                        })}
                    />

                    <div className="classification-analysis-workspace">
                        {activePanel === 'clean' && (
                            <section className="classification-analysis-mode">
                                <header className="classification-mode-heading">
                                    <div>
                                        <span className="evaluation-eyebrow">Inferencia original</span>
                                        <h3>Detecciones de la imagen</h3>
                                        <p>
                                            Inspecciona las cajas, clases y confianzas del detector antes de aplicar perturbaciones.
                                        </p>
                                    </div>
                                    {cleanDetection && !loadingPrediction && (
                                        <button
                                            type="button"
                                            className="secondary-action"
                                            onClick={handlePredict}
                                            disabled={busy}
                                        >
                                            Repetir detección
                                        </button>
                                    )}
                                </header>

                                {loadingPrediction ? (
                                    <ClassificationLoadingState
                                        title="Ejecutando detección original"
                                        description="Las cajas y confianzas aparecerán en este mismo espacio de análisis."
                                        onCancel={handleCancelOperation}
                                    />
                                ) : cleanDetection ? (
                                    <CleanDetectionResult
                                        detection={cleanDetection}
                                        selectedImage={selectedImage}
                                    />
                                ) : (
                                    <AnalysisPrompt
                                        eyebrow="Detección original"
                                        title="Detecta los objetos de esta imagen"
                                        description="Obtén las cajas, clases y confianzas sin modificar la imagen seleccionada."
                                        actionLabel="Ejecutar detección"
                                        onAction={handlePredict}
                                    />
                                )}
                            </section>
                        )}

                        {activePanel === 'attack' && (
                            <section className="classification-analysis-mode">
                                <header className="classification-mode-heading">
                                    <div>
                                        <span className="evaluation-eyebrow">Comparación adversaria</span>
                                        <h3>Detecciones antes y después del ataque</h3>
                                        <p>
                                            Conserva el visor sincronizado, la superposición, las cajas y Eigen-CAM para comparar ambas condiciones.
                                        </p>
                                    </div>
                                    {attackResult && !loading && (
                                        <button
                                            type="button"
                                            className="secondary-action"
                                            onClick={handleRunAttack}
                                            disabled={busy || !isAttackSupported}
                                        >
                                            Repetir ataque
                                        </button>
                                    )}
                                </header>

                                {loading ? (
                                    <ClassificationLoadingState
                                        title={`Aplicando ${attackName}`}
                                        description="Se calculan la referencia original, la imagen adversaria, las detecciones y, cuando el detector lo permite, Eigen-CAM."
                                        onCancel={handleCancelOperation}
                                    />
                                ) : attackResult ? (
                                    <>
                                        <AdversarialImageExplorer
                                            key={attackResult.result_id}
                                            taskType={taskType}
                                            selectedImage={selectedImage}
                                            attackResult={attackResult}
                                            filename={selectedImage.filename}
                                            imageDimensions={imageDimensions}
                                            perturbationMetrics={{
                                                linf: attackResult.comparison.perturbation_linf,
                                                l2: attackResult.comparison.perturbation_l2,
                                            }}
                                        />
                                        <DetectionResults result={attackResult} />
                                    </>
                                ) : (
                                    <AnalysisPrompt
                                        eyebrow="Robustez adversaria"
                                        title="Compara las detecciones frente al ataque"
                                        description={`El ataque ${attackName} calculará también la detección original y los artefactos visuales disponibles para este modelo.`}
                                        actionLabel={`Aplicar ${attackName} y comparar`}
                                        onAction={handleRunAttack}
                                        disabled={!isAttackSupported}
                                    />
                                )}
                            </section>
                        )}

                        {activePanel === 'explanation' && (
                            <section className="classification-analysis-mode">
                                <header className="classification-mode-heading">
                                    <div>
                                        <span className="evaluation-eyebrow">Explicabilidad</span>
                                        <h3>Dónde se fija el detector</h3>
                                        <p>
                                            Analiza Eigen-CAM como resultado independiente o compara la atención original y adversaria cuando el ataque ya la ha generado.
                                        </p>
                                    </div>
                                    {detectionExplanation?.mode === 'single' &&
                                        !loadingExplainability && (
                                        <button
                                            type="button"
                                            className="secondary-action"
                                            onClick={handleGenerateExplainability}
                                            disabled={busy}
                                        >
                                            Regenerar Eigen-CAM
                                        </button>
                                    )}
                                </header>

                                {loadingExplainability ? (
                                    <ClassificationLoadingState
                                        title="Generando Eigen-CAM"
                                        description="El mapa resaltará las regiones con mayor influencia sobre las detecciones originales."
                                        onCancel={handleCancelOperation}
                                    />
                                ) : detectionExplanation ? (
                                    <DetectionExplainabilityResult
                                        explanation={detectionExplanation}
                                        selectedImage={selectedImage}
                                    />
                                ) : (
                                    <AnalysisPrompt
                                        eyebrow="Atención visual"
                                        title="Genera Eigen-CAM para esta imagen"
                                        description="Obtén una explicación visual de las detecciones sin necesidad de ejecutar un ataque."
                                        actionLabel="Generar Eigen-CAM"
                                        onAction={handleGenerateExplainability}
                                    />
                                )}
                            </section>
                        )}
                    </div>
                </>
            )}
        </section>
    )
}

export default ComparisonView
