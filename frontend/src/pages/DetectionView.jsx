import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { attackDetectionImage, detectImage } from '../api/detect'
import { useExperiment } from '../context/ExperimentContext'
import { generateDetectionExplainability } from '../api/explainability'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function DetectionTable({ detections }) {
    if (!detections || detections.length === 0) {
        return <p>No hay detecciones.</p>
    }

    return (
        <div className="evaluation-table-wrapper">
            <table className="evaluation-table">
                <thead>
                    <tr>
                        <th>Clase</th>
                        <th>Confianza</th>
                        <th>x1</th>
                        <th>y1</th>
                        <th>x2</th>
                        <th>y2</th>
                    </tr>
                </thead>

                <tbody>
                    {detections.map((detection, index) => (
                        <tr key={`${detection.label}-${index}`}>
                            <td>{detection.label}</td>
                            <td>{detection.confidence}</td>
                            <td>{detection.bbox.x1}</td>
                            <td>{detection.bbox.y1}</td>
                            <td>{detection.bbox.x2}</td>
                            <td>{detection.bbox.y2}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

function DetectionView() {
    const navigate = useNavigate()

    const {
        selectedDataset,
        selectedImage,
        activeModel,
        selectedAttack,
        attackParameters,
    } = useExperiment()

    const [detectionResult, setDetectionResult] = useState(null)
    const [attackResult, setAttackResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [attackLoading, setAttackLoading] = useState(false)
    const [error, setError] = useState('')
    const [explainabilityResult, setExplainabilityResult] = useState(null)
    const [explainabilityLoading, setExplainabilityLoading] = useState(false)

    useEffect(() => {
        setDetectionResult(null)
        setAttackResult(null)
        setError('')
        setExplainabilityResult(null)
    }, [selectedDataset, selectedImage, activeModel])

    const detectionAttackSupported =
        selectedAttack === 'random_noise' || selectedAttack === 'yolo_confidence'

    async function handleDetect() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        if (activeModel?.task_type !== 'detection') {
            setError('Debes seleccionar un modelo de detección, como YOLO.')
            return
        }

        try {
            setLoading(true)
            setError('')
            setDetectionResult(null)
            setExplainabilityResult(null)

            const result = await detectImage({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
            })

            setDetectionResult(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    async function handleDetectionAttack() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        if (activeModel?.task_type !== 'detection') {
            setError('Debes seleccionar un modelo de detección, como YOLO.')
            return
        }

        if (!detectionAttackSupported) {
            setError(
                'Selecciona Random Noise o YOLO Confidence para atacar un modelo de detección.',
            )
            return
        }

        try {
            setAttackLoading(true)
            setError('')
            setAttackResult(null)

            const result = await attackDetectionImage({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
                attackType: selectedAttack,
                epsilon: attackParameters.epsilon,
                alpha: attackParameters.alpha,
                steps: attackParameters.steps,
            })

            setAttackResult(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setAttackLoading(false)
        }
    }

    async function handleGenerateDetectionHeatmap() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        if (activeModel?.task_type !== 'detection') {
            setError('Debes seleccionar un modelo de detección, como YOLO.')
            return
        }

        try {
            setExplainabilityLoading(true)
            setError('')
            setExplainabilityResult(null)

            const result = await generateDetectionExplainability({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
            })

            setExplainabilityResult(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setExplainabilityLoading(false)
        }
    }

    return (
        <section className="detection-page">
            <div className="comparison-header">
                <div>
                    <h2>Detección de Objetos</h2>
                    <p>
                        Ejecuta un modelo de detección sobre la imagen seleccionada y
                        visualiza las cajas detectadas.
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

            <div className="context-summary">
                <h3>Resumen del Contexto</h3>

                <div className="context-grid">
                    <p>
                        <strong>Dataset:</strong>{' '}
                        {selectedDataset?.name || 'Sin dataset seleccionado'}
                    </p>

                    <p>
                        <strong>Imagen:</strong>{' '}
                        {selectedImage?.filename || 'Sin imagen seleccionada'}
                    </p>

                    <p>
                        <strong>Modelo:</strong>{' '}
                        {activeModel?.name || 'Sin modelo activo'}
                    </p>

                    <p>
                        <strong>Tipo de modelo:</strong>{' '}
                        {activeModel?.task_type || '-'}
                    </p>

                    <p>
                        <strong>Ataque seleccionado:</strong>{' '}
                        {getAttackName(selectedAttack)}
                    </p>

                    <p>
                        <strong>Parámetros:</strong>{' '}
                        {formatAttackParameters(selectedAttack, attackParameters)}
                    </p>
                </div>
            </div>

            {error && <p className="error">{error}</p>}

            {!selectedDataset || !selectedImage ? (
                <div className="empty-state">
                    <h3>No hay imagen seleccionada</h3>
                    <p>
                        Vuelve al Dashboard y selecciona un dataset y una imagen antes
                        de ejecutar la detección.
                    </p>

                    <button
                        type="button"
                        className="secondary-action"
                        onClick={() => navigate('/dashboard')}
                    >
                        Volver al Dashboard
                    </button>
                </div>
            ) : activeModel?.task_type !== 'detection' ? (
                <div className="empty-state">
                    <h3>Modelo no compatible</h3>
                    <p>
                        La vista de detección requiere un modelo de detección.
                        Selecciona YOLO en el Dashboard.
                    </p>

                    <button
                        type="button"
                        className="secondary-action"
                        onClick={() => navigate('/dashboard')}
                    >
                        Seleccionar modelo
                    </button>
                </div>
            ) : (
                <>
                    <div className="detection-actions">
                        <button
                            type="button"
                            className="primary-action"
                            onClick={handleDetect}
                            disabled={loading}
                        >
                            {loading ? 'Detectando...' : 'Ejecutar detección'}
                        </button>

                        <button
                            type="button"
                            className="secondary-action"
                            onClick={handleDetectionAttack}
                            disabled={attackLoading || !detectionAttackSupported}
                            title={
                                detectionAttackSupported
                                    ? 'Aplicar ataque al detector'
                                    : 'Selecciona Random Noise o YOLO Confidence'
                            }
                        >
                            {attackLoading
                                ? 'Aplicando ataque...'
                                : 'Aplicar ataque de detección'}
                        </button>

                        <button
                            type="button"
                            className="secondary-action"
                            onClick={handleGenerateDetectionHeatmap}
                            disabled={explainabilityLoading}
                        >
                            {explainabilityLoading ? 'Generando heatmap...' : 'Generar heatmap'}
                        </button>
                    </div>

                    {!detectionAttackSupported && (
                        <p className="muted-text">
                            Para modelos de detección puedes usar Random Noise o YOLO
                            Confidence.
                        </p>
                    )}

                    <div className="detection-grid">
                        <article className="analysis-card">
                            <h3>Imagen original</h3>

                            <img
                                className="analysis-image"
                                src={`${API_BASE_URL}${selectedImage.url}`}
                                alt={selectedImage.filename}
                            />
                        </article>

                        <article className="analysis-card">
                            <h3>Imagen anotada</h3>

                            {detectionResult?.annotated_image_url ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${detectionResult.annotated_image_url}`}
                                    alt="Detección anotada"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Ejecuta la detección para ver las cajas.
                                </div>
                            )}
                        </article>
                    </div>

                    <section className="comparison-detail-card">
                        <h3>Detecciones</h3>

                        {!detectionResult ? (
                            <p>No se ha ejecutado ninguna detección todavía.</p>
                        ) : (
                            <DetectionTable detections={detectionResult.detections} />
                        )}
                    </section>

                    {explainabilityResult && (
                        <section className="comparison-detail-card">
                            <h3>Explicabilidad visual de detección</h3>

                            <p>
                                Tipo de explicación:{' '}
                                <strong>{explainabilityResult.explanation_type}</strong>
                            </p>

                            <div className="detection-explainability-grid">
                                <article className="analysis-card">
                                    <h3>EigenCAM + detecciones</h3>

                                    <img
                                        className="analysis-image"
                                        src={`${API_BASE_URL}${explainabilityResult.explanation_image_url}`}
                                        alt="Explicabilidad de detección"
                                    />
                                </article>

                                <article className="analysis-card">
                                    <h3>Detecciones explicadas</h3>

                                    <DetectionTable detections={explainabilityResult.detections} />
                                </article>
                            </div>
                        </section>
                    )}

                    {attackResult && (
                        <>
                            <section className="comparison-detail-card">
                                <h3>Comparación adversaria de detección</h3>

                                <div className="detection-attack-grid">
                                    <article className="analysis-card">
                                        <h3>Original anotada</h3>
                                        <img
                                            className="analysis-image"
                                            src={`${API_BASE_URL}${attackResult.original_annotated_image_url}`}
                                            alt="Original anotada"
                                        />
                                    </article>

                                    <article className="analysis-card">
                                        <h3>Perturbación</h3>
                                        <img
                                            className="analysis-image"
                                            src={`${API_BASE_URL}${attackResult.perturbation_image_url}`}
                                            alt="Perturbación"
                                        />
                                    </article>

                                    <article className="analysis-card">
                                        <h3>Imagen adversaria</h3>
                                        <img
                                            className="analysis-image"
                                            src={`${API_BASE_URL}${attackResult.adversarial_image_url}`}
                                            alt="Imagen adversaria"
                                        />
                                    </article>

                                    <article className="analysis-card">
                                        <h3>Adversaria anotada</h3>
                                        <img
                                            className="analysis-image"
                                            src={`${API_BASE_URL}${attackResult.adversarial_annotated_image_url}`}
                                            alt="Adversaria anotada"
                                        />
                                    </article>

                                    <article className="analysis-card">
                                        <h3>EigenCAM original</h3>
                                        <img
                                            className="analysis-image"
                                            src={`${API_BASE_URL}${attackResult.original_explanation_image_url}`}
                                            alt="EigenCAM original"
                                        />
                                    </article>

                                    <article className="analysis-card">
                                        <h3>EigenCAM adversaria</h3>
                                        <img
                                            className="analysis-image"
                                            src={`${API_BASE_URL}${attackResult.adversarial_explanation_image_url}`}
                                            alt="EigenCAM adversaria"
                                        />
                                    </article>
                                </div>
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Métricas del ataque de detección</h3>

                                <div className="metric-grid">
                                    <article className="metric-card">
                                        <span>Detecciones originales</span>
                                        <strong>
                                            {
                                                attackResult.comparison
                                                    .original_detection_count
                                            }
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>Detecciones adversarias</span>
                                        <strong>
                                            {
                                                attackResult.comparison
                                                    .adversarial_detection_count
                                            }
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>
                                            Diferencia de detecciones (original −
                                            adversaria)
                                        </span>
                                        <strong>
                                            {
                                                attackResult.comparison
                                                    .detections_difference
                                            }
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>Confianza media original</span>
                                        <strong>
                                            {
                                                attackResult.comparison
                                                    .average_original_confidence
                                            }
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>Confianza media adversaria</span>
                                        <strong>
                                            {
                                                attackResult.comparison
                                                    .average_adversarial_confidence
                                            }
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>
                                            Diferencia de confianza (original −
                                            adversaria)
                                        </span>
                                        <strong>
                                            {
                                                attackResult.comparison
                                                    .average_confidence_difference
                                            }
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>Resultado legacy del ataque</span>
                                        <strong>
                                            {attackResult.comparison.attack_success
                                                ? 'Exitoso'
                                                : 'No exitoso'}
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>Perturbación L∞</span>
                                        <strong>
                                            {attackResult.comparison.perturbation_linf}
                                        </strong>
                                    </article>

                                    <article className="metric-card">
                                        <span>Perturbación L2</span>
                                        <strong>
                                            {attackResult.comparison.perturbation_l2}
                                        </strong>
                                    </article>
                                </div>

                                <p className="muted-text">
                                    Una diferencia positiva de confianza indica una
                                    caída tras el ataque. Una diferencia positiva de
                                    detecciones indica que hay menos detecciones
                                    adversarias, pero no mide calidad sin ground truth.
                                </p>
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Detecciones originales vs adversarias</h3>

                                <div className="detection-comparison-tables">
                                    <div>
                                        <h4>Originales</h4>
                                        <DetectionTable
                                            detections={
                                                attackResult.original_detections
                                            }
                                        />
                                    </div>

                                    <div>
                                        <h4>Adversarias</h4>
                                        <DetectionTable
                                            detections={
                                                attackResult.adversarial_detections
                                            }
                                        />
                                    </div>
                                </div>
                            </section>
                        </>
                    )}
                </>
            )}
        </section>
    )
}

export default DetectionView
