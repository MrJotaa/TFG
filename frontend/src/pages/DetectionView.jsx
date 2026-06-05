import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { detectImage } from '../api/detect'
import { useExperiment } from '../context/ExperimentContext'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function DetectionView() {
    const navigate = useNavigate()

    const {
        selectedDataset,
        selectedImage,
        activeModel,
    } = useExperiment()

    const [detectionResult, setDetectionResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        setDetectionResult(null)
        setError('')
    }, [selectedDataset, selectedImage, activeModel])

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
                        La vista de detección requiere un modelo de detección. Selecciona
                        YOLO en el Dashboard.
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
                    </div>

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
                                    Ejecuta la detección para ver las cajas
                                </div>
                            )}
                        </article>
                    </div>

                    <section className="comparison-detail-card">
                        <h3>Detecciones</h3>

                        {!detectionResult ? (
                            <p>No se ha ejecutado ninguna detección todavía.</p>
                        ) : detectionResult.detections.length === 0 ? (
                            <p>No se han detectado objetos en la imagen.</p>
                        ) : (
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
                                        {detectionResult.detections.map((detection, index) => (
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
                        )}
                    </section>
                </>
            )}
        </section>
    )
}

export default DetectionView