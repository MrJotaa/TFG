import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { generateGradCam } from '../api/explainability'
import { predictImage } from '../api/predict'
import { useExperiment } from '../context/ExperimentContext'

const API_BASE_URL = 'http://127.0.0.1:8000'

const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
}

function PredictionView() {
    const navigate = useNavigate()

    const {
        selectedDataset,
        selectedImage,
        activeModel,
        selectedAttack,
        attackParameters,
    } = useExperiment()

    const [prediction, setPrediction] = useState(null)
    const [gradCam, setGradCam] = useState(null)
    const [loadingPrediction, setLoadingPrediction] = useState(false)
    const [loadingGradCam, setLoadingGradCam] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        setPrediction(null)
        setGradCam(null)
        setError('')
    }, [selectedDataset, selectedImage, activeModel])

    async function handlePredict() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        try {
            setLoadingPrediction(true)
            setError('')
            setPrediction(null)

            const result = await predictImage({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
            })

            setPrediction(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoadingPrediction(false)
        }
    }

    async function handleGenerateGradCam() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        try {
            setLoadingGradCam(true)
            setError('')
            setGradCam(null)

            const result = await generateGradCam({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
            })

            setGradCam(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoadingGradCam(false)
        }
    }

    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack

    return (
        <section className="prediction-page">
            <div className="context-summary">
                <h2>Resumen del Contexto</h2>

                <div className="context-grid">
                    <p>
                        <strong>Dataset:</strong>{' '}
                        {selectedDataset?.name || 'No dataset selected'}
                    </p>

                    <p>
                        <strong>Imagen:</strong>{' '}
                        {selectedImage?.filename || 'No image selected'}
                    </p>

                    <p>
                        <strong>Modelo:</strong>{' '}
                        {activeModel?.name || 'No active model'}
                    </p>

                    <p>
                        <strong>Estado del Ataque:</strong>{' '}
                        <span className="status-success">No Atacado</span>
                    </p>

                    <p>
                        <strong>Ground truth:</strong>{' '}
                        {selectedImage?.label || 'No label'}
                    </p>
                </div>
            </div>

            {error && <p className="error">{error}</p>}

            {!selectedDataset || !selectedImage ? (
                <div className="empty-state">
                    <h3>No hay imagen seleccionada</h3>
                    <p>
                        Vuelve al dashboard y selecciona un dataset y una imagen antes
                        de ejecutar la predicción.
                    </p>

                    <button
                        type="button"
                        className="secondary-action"
                        onClick={() => navigate('/dashboard')}
                    >
                        Volver al Dashboard
                    </button>
                </div>
            ) : (
                <>
                    <div className="prediction-grid">
                        <article className="analysis-card">
                            <h3>Imagen Original</h3>

                            <img
                                className="analysis-image"
                                src={`${API_BASE_URL}${selectedImage.url}`}
                                alt={selectedImage.filename}
                            />
                        </article>

                        <article className="analysis-card">
                            <h3>Grad-CAM / Heatmap</h3>

                            {gradCam ? (
                                <>
                                    <img
                                        className="analysis-image"
                                        src={`${API_BASE_URL}${gradCam.gradcam_image_url}`}
                                        alt="Grad-CAM"
                                    />

                                    <p>
                                        Clase explicada:{' '}
                                        <strong>{gradCam.prediction}</strong>
                                    </p>
                                </>
                            ) : (
                                <>
                                    <div className="image-placeholder">
                                        Heatmap no generado
                                    </div>

                                    <button
                                        type="button"
                                        className="secondary-action"
                                        onClick={handleGenerateGradCam}
                                        disabled={loadingGradCam}
                                    >
                                        {loadingGradCam
                                            ? 'Generando Grad-CAM...'
                                            : 'Generar Grad-CAM'}
                                    </button>
                                </>
                            )}
                        </article>

                        <article className="analysis-card prediction-main-card">
                            <h3>Predicción Principal</h3>

                            {prediction ? (
                                <>
                                    <p className="main-prediction">
                                        Clase Predicha:{' '}
                                        <strong>{prediction.prediction}</strong>
                                    </p>
                                    <p>
                                        Confianza:{' '}
                                        <strong>{prediction.confidence}</strong>
                                    </p>
                                </>
                            ) : (
                                <p>No se ha ejecutado ninguna predicción aún.</p>
                            )}

                            <button
                                type="button"
                                className="primary-action"
                                onClick={handlePredict}
                                disabled={loadingPrediction}
                            >
                                {loadingPrediction
                                    ? 'Prediciendo...'
                                    : 'Ejecutar Predicción'}
                            </button>
                        </article>

                        <article className="analysis-card">
                            <h3>Predicciones Top-K</h3>

                            {prediction ? (
                                <div className="topk-list">
                                    {prediction.top_k.map((item) => (
                                        <div
                                            key={item.label}
                                            className="topk-row"
                                        >
                                            <span>{item.label}</span>
                                            <strong>{item.probability}</strong>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p>No hay resultados disponibles para las predicciones Top-K.</p>
                            )}
                        </article>
                    </div>

                    <div className="analysis-footer">
                        <div className="attack-summary-line">
                            <strong>Nombre del Ataque:</strong> {attackName}{' '}
                            <strong>Parámetros del Ataque:</strong> ε=
                            {attackParameters.epsilon}
                            {selectedAttack === 'pgd' &&
                                ` · α=${attackParameters.alpha} · steps=${attackParameters.steps}`}
                        </div>

                        <div className="analysis-actions">
                            <button
                                type="button"
                                className="secondary-action"
                                onClick={() => navigate('/dashboard')}
                            >
                                Volver al Dashboard
                            </button>

                            <button
                                type="button"
                                className="primary-action"
                                onClick={() => navigate('/comparison')}
                            >
                                Aplicar Ataque Adversario
                            </button>
                        </div>
                    </div>
                </>
            )}
        </section>
    )
}

export default PredictionView