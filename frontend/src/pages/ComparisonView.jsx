import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { applyAttack } from '../api/attack'
import { useExperiment } from '../context/ExperimentContext'

const API_BASE_URL = 'http://127.0.0.1:8000'

const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
}

function ComparisonView() {
    const navigate = useNavigate()

    const {
        selectedDataset,
        selectedImage,
        activeModel,
        selectedAttack,
        attackParameters,
    } = useExperiment()

    const [attackResult, setAttackResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        setAttackResult(null)
        setError('')
    }, [selectedDataset, selectedImage, activeModel, selectedAttack, attackParameters])

    async function handleRunAttack() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        try {
            setLoading(true)
            setError('')
            setAttackResult(null)

            const result = await applyAttack({
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
            setLoading(false)
        }
    }

    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack
    const comparison = attackResult?.comparison

    return (
        <section className="comparison-page">
            <div className="comparison-header">
                <div>
                    <h2>Adversarial Comparison and Analysis</h2>
                    <p>
                        Compara la predicción original frente a la predicción tras aplicar
                        un ataque adversario.
                    </p>
                </div>

                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard')}
                >
                    Back to Dashboard
                </button>
            </div>

            <div className="context-summary">
                <h3>Context Summary</h3>

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
                        <strong>Attack:</strong> {attackName}
                    </p>

                    <p>
                        <strong>Parameters:</strong> ε={attackParameters.epsilon}
                        {selectedAttack === 'pgd' &&
                            ` · α=${attackParameters.alpha} · steps=${attackParameters.steps}`}
                    </p>

                    <p>
                        <strong>Attack Status:</strong>{' '}
                        {attackResult ? (
                            <span className="status-danger">Attacked</span>
                        ) : (
                            <span className="status-muted">Not executed</span>
                        )}
                    </p>
                </div>
            </div>

            {error && <p className="error">{error}</p>}

            {!selectedDataset || !selectedImage ? (
                <div className="empty-state">
                    <h3>No hay imagen seleccionada</h3>
                    <p>
                        Vuelve al dashboard y selecciona un dataset y una imagen antes
                        de ejecutar la comparación adversaria.
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
                    <div className="comparison-actions-top">
                        <button
                            type="button"
                            className="primary-action"
                            onClick={handleRunAttack}
                            disabled={loading}
                        >
                            {loading ? 'Running attack...' : 'Run Attack'}
                        </button>
                    </div>

                    <div className="comparison-visual-grid">
                        <article className="analysis-card">
                            <h3>Original Image</h3>
                            <img
                                className="analysis-image"
                                src={`${API_BASE_URL}${selectedImage.url}`}
                                alt={selectedImage.filename}
                            />
                        </article>

                        <article className="analysis-card perturbation-card">
                            <h3>Perturbation Map</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.perturbation_image_url}`}
                                    alt="Perturbation map"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Perturbation not generated
                                </div>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Adversarial Image</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.adversarial_image_url}`}
                                    alt="Adversarial"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Adversarial image not generated
                                </div>
                            )}
                        </article>
                    </div>

                    <div className="comparison-grid">
                        <article className="analysis-card">
                            <h3>Original Grad-CAM</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.original_gradcam_image_url}`}
                                    alt="Original Grad-CAM"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Original heatmap not generated
                                </div>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Adversarial Grad-CAM</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.adversarial_gradcam_image_url}`}
                                    alt="Adversarial Grad-CAM"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Adversarial heatmap not generated
                                </div>
                            )}
                        </article>

                        <article className="analysis-card prediction-main-card">
                            <h3>Original Prediction</h3>

                            {attackResult ? (
                                <>
                                    <p className="main-prediction">
                                        Predicted Class:{' '}
                                        <strong>
                                            {attackResult.original_prediction.prediction}
                                        </strong>
                                    </p>
                                    <p>
                                        Confidence:{' '}
                                        <strong>
                                            {attackResult.original_prediction.confidence}
                                        </strong>
                                    </p>
                                </>
                            ) : (
                                <p>No original prediction available.</p>
                            )}
                        </article>

                        <article className="analysis-card prediction-main-card">
                            <h3>Attacked Prediction</h3>

                            {attackResult ? (
                                <>
                                    <p className="main-prediction danger-text">
                                        Predicted Class:{' '}
                                        <strong>
                                            {
                                                attackResult.adversarial_prediction
                                                    .prediction
                                            }
                                        </strong>
                                    </p>
                                    <p>
                                        Confidence:{' '}
                                        <strong>
                                            {
                                                attackResult.adversarial_prediction
                                                    .confidence
                                            }
                                        </strong>
                                    </p>
                                </>
                            ) : (
                                <p>No adversarial prediction available.</p>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Original Top-K</h3>

                            {attackResult ? (
                                <div className="topk-list">
                                    {attackResult.original_prediction.top_k.map(
                                        (item) => (
                                            <div
                                                key={item.label}
                                                className="topk-row"
                                            >
                                                <span>{item.label}</span>
                                                <strong>{item.probability}</strong>
                                            </div>
                                        ),
                                    )}
                                </div>
                            ) : (
                                <p>No Top-K results available.</p>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Attacked Top-K</h3>

                            {attackResult ? (
                                <div className="topk-list">
                                    {attackResult.adversarial_prediction.top_k.map(
                                        (item) => (
                                            <div
                                                key={item.label}
                                                className="topk-row"
                                            >
                                                <span>{item.label}</span>
                                                <strong>{item.probability}</strong>
                                            </div>
                                        ),
                                    )}
                                </div>
                            ) : (
                                <p>No Top-K results available.</p>
                            )}
                        </article>
                    </div>

                    <div className="comparison-detail-card">
                        <h3>Attack Details</h3>

                        {attackResult ? (
                            <div className="details-grid">
                                <p>
                                    <strong>Attack Name:</strong> {attackName}
                                </p>
                                <p>
                                    <strong>Epsilon:</strong>{' '}
                                    {attackResult.parameters.epsilon}
                                </p>
                                {selectedAttack === 'pgd' && (
                                    <>
                                        <p>
                                            <strong>Alpha:</strong>{' '}
                                            {attackResult.parameters.alpha}
                                        </p>
                                        <p>
                                            <strong>Steps:</strong>{' '}
                                            {attackResult.parameters.steps}
                                        </p>
                                    </>
                                )}
                                <p>
                                    <strong>L∞ perturbation:</strong>{' '}
                                    {comparison.perturbation_linf}
                                </p>
                                <p>
                                    <strong>L2 perturbation:</strong>{' '}
                                    {comparison.perturbation_l2}
                                </p>
                                <p>
                                    <strong>Attack Result:</strong>{' '}
                                    {comparison.attack_success ? (
                                        <span className="status-danger">
                                            Successful
                                        </span>
                                    ) : (
                                        <span className="status-success">
                                            Not successful
                                        </span>
                                    )}
                                </p>
                            </div>
                        ) : (
                            <p>Run an attack to see details.</p>
                        )}
                    </div>

                    <div className="comparison-detail-card">
                        <h3>Comparative Analysis</h3>

                        {attackResult ? (
                            <div className="details-grid">
                                <p>
                                    <strong>Prediction Changed:</strong>{' '}
                                    {comparison.prediction_changed ? (
                                        <span className="status-danger">
                                            Yes ({comparison.original_label} →{' '}
                                            {comparison.adversarial_label})
                                        </span>
                                    ) : (
                                        <span className="status-success">No</span>
                                    )}
                                </p>

                                <p>
                                    <strong>Confidence Difference:</strong>{' '}
                                    {comparison.confidence_difference}
                                </p>

                                <p>
                                    <strong>Original Confidence:</strong>{' '}
                                    {comparison.original_confidence}
                                </p>

                                <p>
                                    <strong>Adversarial Confidence:</strong>{' '}
                                    {comparison.adversarial_confidence}
                                </p>

                                <p>
                                    <strong>Attack Impact:</strong>{' '}
                                    {comparison.attack_success
                                        ? 'The attack changed the model prediction.'
                                        : 'The attack did not change the model prediction.'}
                                </p>
                            </div>
                        ) : (
                            <p>Run an attack to see the comparative analysis.</p>
                        )}
                    </div>

                    <div className="analysis-actions comparison-bottom-actions">
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={() => navigate('/dashboard')}
                        >
                            Back
                        </button>

                        <button
                            type="button"
                            className="primary-action"
                            onClick={handleRunAttack}
                            disabled={loading}
                        >
                            {loading ? 'Running...' : 'Run New Attack'}
                        </button>
                    </div>
                </>
            )}
        </section>
    )
}

export default ComparisonView