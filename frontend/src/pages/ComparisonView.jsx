import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { applyAttack } from '../api/attack'
import { isRequestCancelled, resolveApiUrl } from '../api/client'
import { attackDetectionImage, detectImage } from '../api/detect'
import { generateDetectionExplainability } from '../api/explainability'
import { predictImage } from '../api/predict'
import AdversarialImageExplorer from '../components/visualExplorer/AdversarialImageExplorer'
import { useExperiment } from '../context/ExperimentContext'
import { resolveCleanDetectionResult } from '../utils/adversarialExplorer'
import {
    attackParametersKey,
    isAttackSupportedForTask,
} from '../utils/attackCatalog'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'

function Metric({ label, value, tone = '' }) {
    return (
        <article className={`metric-card ${tone}`}>
            <span>{label}</span>
            <strong>{value ?? '—'}</strong>
        </article>
    )
}

function DetectionList({ detections, title }) {
    return (
        <details className="adversarial-detection-list">
            <summary>
                {title} <span>{detections?.length ?? 0}</span>
            </summary>
            {detections?.length ? (
                <div className="evaluation-table-wrapper">
                    <table className="evaluation-table">
                        <thead>
                            <tr>
                                <th>Clase</th>
                                <th>Confianza</th>
                                <th>Bounding box</th>
                            </tr>
                        </thead>
                        <tbody>
                            {detections.map((detection, index) => (
                                <tr key={`${detection.label}-${index}`}>
                                    <td>{detection.label}</td>
                                    <td>{detection.confidence}</td>
                                    <td>
                                        {detection.bbox.x1}, {detection.bbox.y1} →{' '}
                                        {detection.bbox.x2}, {detection.bbox.y2}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <p>No hay detecciones en esta condición.</p>
            )}
        </details>
    )
}

function ClassificationResults({ result }) {
    const comparison = result.comparison

    return (
        <>
            <div className="metric-grid">
                <Metric
                    label="Predicción original"
                    value={result.original_prediction.prediction}
                />
                <Metric
                    label="Predicción adversaria"
                    value={result.adversarial_prediction.prediction}
                    tone={comparison.prediction_changed ? 'tone-danger' : ''}
                />
                <Metric
                    label="Confianza original"
                    value={comparison.original_confidence}
                />
                <Metric
                    label="Confianza adversaria"
                    value={comparison.adversarial_confidence}
                />
                <Metric
                    label="Cambio de confianza"
                    value={comparison.confidence_difference}
                />
                <Metric
                    label="Resultado del ataque"
                    value={comparison.attack_success ? 'Exitoso' : 'No exitoso'}
                    tone={comparison.attack_success ? 'tone-danger' : 'tone-success'}
                />
            </div>

            <div className="adversarial-topk-grid">
                {[
                    ['Top-K original', result.original_prediction.top_k],
                    ['Top-K adversarial', result.adversarial_prediction.top_k],
                ].map(([title, predictions]) => (
                    <section key={title} className="analysis-card">
                        <h4>{title}</h4>
                        <div className="topk-list">
                            {predictions.map((prediction) => (
                                <div key={prediction.label} className="topk-row">
                                    <span>{prediction.label}</span>
                                    <strong>{prediction.probability}</strong>
                                </div>
                            ))}
                        </div>
                    </section>
                ))}
            </div>
        </>
    )
}

function CleanPredictionResult({ prediction, selectedImage }) {
    return (
        <section className="clean-prediction-result" aria-live="polite">
            <div className="clean-prediction-image">
                <span className="evaluation-eyebrow">Imagen seleccionada</span>
                <img
                    src={resolveApiUrl(selectedImage.url)}
                    alt={selectedImage.filename}
                />
            </div>

            <div className="clean-prediction-summary">
                <span className="evaluation-eyebrow">Predicción limpia</span>
                <h3>{prediction.prediction}</h3>
                <p className="clean-prediction-confidence">
                    Confianza <strong>{prediction.confidence}</strong>
                </p>

                <div className="topk-list">
                    {prediction.top_k?.map((item) => (
                        <div key={item.label} className="topk-row">
                            <span>{item.label}</span>
                            <strong>{item.probability}</strong>
                        </div>
                    ))}
                </div>

                <p className="muted-text">
                    Aplica el ataque para comparar esta referencia con el resultado
                    adversario sin abandonar el análisis individual.
                </p>
            </div>
        </section>
    )
}

function CleanDetectionResult({ detection, selectedImage }) {
    return (
        <section className="clean-detection-result" aria-live="polite">
            <div className="clean-detection-image">
                <span className="evaluation-eyebrow">Detección limpia</span>
                <img
                    src={resolveApiUrl(
                        detection.annotated_image_url ?? selectedImage.url,
                    )}
                    alt={`Detecciones sobre ${selectedImage.filename}`}
                />
            </div>

            <div className="clean-detection-summary">
                <span className="evaluation-eyebrow">Objetos detectados</span>
                <h3>{detection.detections?.length ?? 0}</h3>
                <p className="muted-text">
                    La imagen conserva su geometría original y las cajas corresponden
                    a la inferencia limpia del modelo.
                </p>
                <DetectionList
                    title="Detalle de detecciones"
                    detections={detection.detections}
                />
            </div>
        </section>
    )
}

function DetectionExplainabilityResult({ result }) {
    return (
        <section className="detection-explainability-result" aria-live="polite">
            <div className="clean-detection-image">
                <span className="evaluation-eyebrow">Explicabilidad visual</span>
                <img
                    src={resolveApiUrl(result.explanation_image_url)}
                    alt="Mapa EigenCAM con detecciones"
                />
            </div>

            <div className="clean-detection-summary">
                <span className="evaluation-eyebrow">Método</span>
                <h3>{result.explanation_type || 'EigenCAM'}</h3>
                <p className="muted-text">
                    El mapa señala las regiones con mayor influencia sobre las
                    detecciones de esta imagen.
                </p>
                <DetectionList
                    title="Detecciones explicadas"
                    detections={result.detections}
                />
            </div>
        </section>
    )
}

function DetectionWorkspaceTabs({ activePanel, availability, onSelect }) {
    const tabs = [
        ['clean', 'Detección', availability.clean],
        ['attack', 'Comparación', availability.attack],
        ['explanation', 'Explicabilidad', availability.explanation],
    ]

    return (
        <nav className="individual-analysis-tabs" aria-label="Resultados del análisis">
            {tabs.map(([panel, label, available]) => (
                <button
                    key={panel}
                    type="button"
                    className={activePanel === panel ? 'active' : ''}
                    aria-current={activePanel === panel ? 'page' : undefined}
                    onClick={() => onSelect(panel)}
                    disabled={!available}
                >
                    {label}
                    <span aria-hidden="true">{available ? '●' : '○'}</span>
                </button>
            ))}
        </nav>
    )
}

function DetectionResults({ result }) {
    const comparison = result.comparison

    return (
        <>
            <div className="metric-grid">
                <Metric
                    label="Detecciones originales"
                    value={comparison.original_detection_count}
                />
                <Metric
                    label="Detecciones adversarias"
                    value={comparison.adversarial_detection_count}
                />
                <Metric
                    label="Caída de detecciones"
                    value={comparison.detections_difference}
                />
                <Metric
                    label="Confianza media original"
                    value={comparison.average_original_confidence}
                />
                <Metric
                    label="Confianza media adversaria"
                    value={comparison.average_adversarial_confidence}
                />
                <Metric
                    label="Caída de confianza"
                    value={comparison.average_confidence_difference}
                />
            </div>

            <p className="muted-text">
                Estas métricas son descriptivas para una imagen. Sin matching contra
                ground truth no equivalen a AP, recall ni supresión científica.
            </p>

            <div className="adversarial-detection-lists">
                <DetectionList
                    title="Detecciones originales"
                    detections={result.original_detections}
                />
                <DetectionList
                    title="Detecciones adversarias"
                    detections={result.adversarial_detections}
                />
            </div>
        </>
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
    const [predictionResultState, setPredictionResultState] = useState(null)
    const [attackResultState, setAttackResultState] = useState(null)
    const [explainabilityResultState, setExplainabilityResultState] = useState(null)
    const [activePanelState, setActivePanelState] = useState(null)
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
        ? requestedPanel
        : requestedPanel === 'attack' && attackResult
            ? 'attack'
            : requestedPanel === 'explanation' && explainabilityResult
                ? 'explanation'
                : cleanDetection
                    ? 'clean'
                    : attackResult
                        ? 'attack'
                        : explainabilityResult
                            ? 'explanation'
                            : 'clean'
    const error = errorState?.contextKey === imageContextKey
        ? errorState.message
        : ''
    const busy = loading || loadingPrediction || loadingExplainability

    useEffect(() => () => requestAbortController.current?.abort(), [])

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
            setCurrentError('Debes seleccionar dataset, imagen y modelo.')
            return
        }

        const controller = beginCancellableRequest()
        try {
            setLoadingPrediction(true)
            setCurrentError('')
            setPredictionResultState(null)
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
            setCurrentError('Debes seleccionar dataset, imagen y modelo.')
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
            const result = isDetection
                ? await attackDetectionImage(request)
                : await applyAttack(request)
            setAttackResultState({ contextKey: attackContextKey, data: result })
            if (isDetection) {
                setActivePanelState({ contextKey: imageContextKey, panel: 'attack' })
            }
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
            setCurrentError('Debes seleccionar dataset, imagen y modelo.')
            return
        }

        const controller = beginCancellableRequest()
        try {
            setLoadingExplainability(true)
            setCurrentError('')
            const result = await generateDetectionExplainability({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
                signal: controller.signal,
            })
            setExplainabilityResultState({
                contextKey: imageContextKey,
                data: result,
            })
            setActivePanelState({
                contextKey: imageContextKey,
                panel: 'explanation',
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

    const imageDimensions = selectedImage?.width && selectedImage?.height
        ? `${selectedImage.width} × ${selectedImage.height} px`
        : null

    return (
        <section className="comparison-page">
            <div className="comparison-header">
                <div>
                    <span className="evaluation-eyebrow">Análisis individual</span>
                    <h2>
                        {isDetection
                            ? 'Detección y exploración adversaria'
                            : 'Predicción y exploración adversaria'}
                    </h2>
                    <p>
                        {isDetection
                            ? 'Compara visualmente las detecciones antes y después del ataque.'
                            : 'Obtén la predicción limpia y compara el efecto del ataque sin cambiar de pantalla.'}
                    </p>
                </div>
                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard')}
                >
                    Volver al Dashboard
                </button>
            </div>

            <div className="context-summary adversarial-context-summary">
                <div className="context-grid">
                    <p><strong>Dataset:</strong> {selectedDataset?.name || 'Sin dataset'}</p>
                    <p><strong>Imagen:</strong> {selectedImage?.filename || 'Sin imagen'}</p>
                    <p><strong>Modelo:</strong> {activeModel?.name || 'Sin modelo'}</p>
                    <p><strong>Ataque:</strong> {getAttackName(selectedAttack, attackCatalog)}</p>
                    <p>
                        <strong>Parámetros:</strong>{' '}
                        {formatAttackParameters(selectedAttack, attackParameters, attackCatalog)}
                    </p>
                    <p>
                        <strong>Geometría:</strong> {imageDimensions || 'No disponible'}
                    </p>
                </div>
            </div>

            {error && <p className="error" role="alert">{error}</p>}

            {!selectedDataset || !selectedImage || !activeModel ? (
                <div className="empty-state">
                    <h3>Falta contexto para explorar</h3>
                    <p>Selecciona un dataset, una imagen y un modelo en el Dashboard.</p>
                    <button
                        type="button"
                        className="secondary-action"
                        onClick={() => navigate('/dashboard')}
                    >
                        Configurar experimento
                    </button>
                </div>
            ) : (
                <>
                    {!isAttackSupported && (
                        <div className="evaluation-warning-banner">
                            El ataque seleccionado no es compatible con este modelo.
                            Cambia el ataque desde el Dashboard.
                        </div>
                    )}

                    <div className="individual-analysis-toolbar">
                        <div className="adversarial-image-navigation">
                            <button
                                type="button"
                                className="secondary-action"
                                disabled={
                                    selectedIndex <= 0 ||
                                    busy
                                }
                                onClick={() => selectRelativeImage(-1)}
                            >
                                ← Imagen anterior
                            </button>
                            <span>
                                {selectedIndex >= 0
                                    ? `${selectedIndex + 1} de ${datasetImages.length}`
                                    : 'Imagen seleccionada'}
                            </span>
                            <button
                                type="button"
                                className="secondary-action"
                                disabled={
                                    selectedIndex < 0 ||
                                    selectedIndex >= datasetImages.length - 1 ||
                                    busy
                                }
                                onClick={() => selectRelativeImage(1)}
                            >
                                Imagen siguiente →
                            </button>
                        </div>

                        <div className="individual-analysis-actions">
                            <button
                                type="button"
                                className="secondary-action"
                                onClick={handlePredict}
                                disabled={busy}
                            >
                                {loadingPrediction
                                    ? isDetection
                                        ? 'Detectando…'
                                        : 'Prediciendo…'
                                    : predictionResult
                                      ? isDetection
                                          ? 'Repetir detección'
                                          : 'Repetir predicción'
                                      : isDetection
                                        ? 'Ejecutar detección'
                                        : 'Predecir imagen'}
                            </button>

                            <button
                                type="button"
                                className="primary-action"
                                onClick={handleRunAttack}
                                disabled={busy || !isAttackSupported}
                            >
                                {loading
                                    ? 'Generando resultado…'
                                    : attackResult
                                      ? 'Repetir ataque'
                                    : 'Aplicar ataque y comparar'}
                            </button>

                            {isDetection && (
                                <button
                                    type="button"
                                    className="secondary-action"
                                    onClick={handleGenerateExplainability}
                                    disabled={busy}
                                >
                                    {loadingExplainability
                                        ? 'Generando EigenCAM…'
                                        : explainabilityResult
                                          ? 'Regenerar EigenCAM'
                                          : 'Generar EigenCAM'}
                                </button>
                            )}

                            {busy && (
                                <button
                                    type="button"
                                    className="secondary-action"
                                    onClick={handleCancelOperation}
                                >
                                    Cancelar
                                </button>
                            )}
                        </div>
                    </div>

                    {busy && (
                        <div className="adversarial-explorer-loading" role="status">
                            <span className="evaluation-loading-spinner" aria-hidden="true" />
                            <div>
                                <strong>
                                    {loading
                                        ? 'Generando artefactos visuales'
                                        : loadingExplainability
                                          ? 'Generando explicabilidad visual'
                                          : isDetection
                                            ? 'Ejecutando detección limpia'
                                            : 'Calculando predicción limpia'}
                                </strong>
                                <p>
                                    {loading
                                        ? 'Ampliar y desplazar después no realiza nuevas peticiones.'
                                        : loadingExplainability
                                          ? 'El mapa aparecerá en la pestaña de explicabilidad.'
                                          : 'El resultado aparecerá en este mismo espacio de análisis.'}
                                </p>
                            </div>
                        </div>
                    )}

                    {isDetection && !busy && (
                        <DetectionWorkspaceTabs
                            activePanel={activePanel}
                            availability={{
                                clean: Boolean(cleanDetection),
                                attack: Boolean(attackResult),
                                explanation: Boolean(explainabilityResult),
                            }}
                            onSelect={(panel) => setActivePanelState({
                                contextKey: imageContextKey,
                                panel,
                            })}
                        />
                    )}

                    {!isDetection && cleanPrediction && !attackResult && !busy && (
                        <CleanPredictionResult
                            prediction={cleanPrediction}
                            selectedImage={selectedImage}
                        />
                    )}

                    {!isDetection &&
                        !cleanPrediction &&
                        !attackResult &&
                        !busy && (
                        <div className="individual-analysis-empty">
                            <strong>Elige el nivel de análisis</strong>
                            <p>
                                Puedes predecir primero la imagen limpia o ejecutar
                                directamente el ataque, que también calcula la
                                referencia original.
                            </p>
                        </div>
                    )}

                    {isDetection && !busy && !cleanDetection && !attackResult &&
                        !explainabilityResult && (
                            <div className="individual-analysis-empty">
                                <strong>Elige el resultado que quieres generar</strong>
                                <p>
                                    Ejecuta la detección limpia, aplica el ataque o
                                    genera EigenCAM. Cada resultado se mostrará en su
                                    propia pestaña sin alargar la página.
                                </p>
                            </div>
                        )}

                    {isDetection && !busy && activePanel === 'clean' &&
                        cleanDetection && (
                            <CleanDetectionResult
                                detection={cleanDetection}
                                selectedImage={selectedImage}
                            />
                        )}

                    {isDetection && !busy && activePanel === 'explanation' &&
                        explainabilityResult && (
                            <DetectionExplainabilityResult
                                result={explainabilityResult}
                            />
                        )}

                    {attackResult && (!isDetection || activePanel === 'attack') && !busy && (
                        <>
                            <AdversarialImageExplorer
                                key={attackResult.result_id}
                                taskType={taskType}
                                selectedImage={selectedImage}
                                attackResult={attackResult}
                                filename={selectedImage.filename}
                                imageDimensions={imageDimensions}
                            />

                            <section className="comparison-detail-card">
                                <h3>Impacto sobre la predicción</h3>
                                {isDetection ? (
                                    <DetectionResults result={attackResult} />
                                ) : (
                                    <ClassificationResults result={attackResult} />
                                )}
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Magnitud del ataque</h3>
                                <div className="details-grid">
                                    <p><strong>L∞:</strong> {attackResult.comparison.perturbation_linf}</p>
                                    <p><strong>L2:</strong> {attackResult.comparison.perturbation_l2}</p>
                                    <p>
                                        <strong>Resultado temporal:</strong>{' '}
                                        {attackResult.result_id}
                                    </p>
                                </div>
                            </section>
                        </>
                    )}
                </>
            )}
        </section>
    )
}

export default ComparisonView
