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
                <h2>Context Summary</h2>

                <div className="context-grid">
                    <p>
                        <strong>Dataset:</strong>{' '}
                        {selectedDataset?.name || 'No dataset selected'}
                    </p>

                    <p>
                        <strong>Image:</strong>{' '}
                        {selectedImage?.filename || 'No image selected'}
                    </p>

                    <p>
                        <strong>Model:</strong>{' '}
                        {activeModel?.name || 'No active model'}
                    </p>

                    <p>
                        <strong>Attack Status:</strong>{' '}
                        <span className="status-success">Not Attacked</span>
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
                        Back to Dashboard
                    </button>
                </div>
            ) : (
                <>
                    <div className="prediction-grid">
                        <article className="analysis-card">
                            <h3>Original Image</h3>

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
                                        Heatmap not generated
                                    </div>

                                    <button
                                        type="button"
                                        className="secondary-action"
                                        onClick={handleGenerateGradCam}
                                        disabled={loadingGradCam}
                                    >
                                        {loadingGradCam
                                            ? 'Generating Grad-CAM...'
                                            : 'Generate Grad-CAM'}
                                    </button>
                                </>
                            )}
                        </article>

                        <article className="analysis-card prediction-main-card">
                            <h3>Main Prediction</h3>

                            {prediction ? (
                                <>
                                    <p className="main-prediction">
                                        Predicted Class:{' '}
                                        <strong>{prediction.prediction}</strong>
                                    </p>
                                    <p>
                                        Confidence:{' '}
                                        <strong>{prediction.confidence}</strong>
                                    </p>
                                </>
                            ) : (
                                <p>No prediction executed yet.</p>
                            )}

                            <button
                                type="button"
                                className="primary-action"
                                onClick={handlePredict}
                                disabled={loadingPrediction}
                            >
                                {loadingPrediction
                                    ? 'Predicting...'
                                    : 'Run Prediction'}
                            </button>
                        </article>

                        <article className="analysis-card">
                            <h3>Top-K Predictions</h3>

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
                                <p>No Top-K results available.</p>
                            )}
                        </article>
                    </div>

                    <div className="analysis-footer">
                        <div className="attack-summary-line">
                            <strong>Attack Name:</strong> {attackName}{' '}
                            <strong>Attack Parameters:</strong> ε=
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
                                Back to Dashboard
                            </button>

                            <button
                                type="button"
                                className="primary-action"
                                onClick={() => navigate('/comparison')}
                            >
                                Apply Attack
                            </button>
                        </div>
                    </div>
                </>
            )}
        </section>
    )
}

export default PredictionView