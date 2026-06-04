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
                    <h2>Análisis Comparativo Adversarial</h2>
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
                    Volver al Dashboard
                </button>
            </div>

            <div className="context-summary">
                <h3>Resumen de Contexto</h3>

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
                        <strong>Ataque:</strong> {attackName}
                    </p>

                    <p>
                        <strong>Parámetros:</strong> ε={attackParameters.epsilon}
                        {selectedAttack === 'pgd' &&
                            ` · α=${attackParameters.alpha} · steps=${attackParameters.steps}`}
                    </p>

                    <p>
                        <strong>Estado del Ataque:</strong>{' '}
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
                        Volver al Dashboard
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
                            {loading ? 'Ejecutando ataque...' : 'Ejecutar Ataque'}
                        </button>
                    </div>

                    <div className="comparison-visual-grid">
                        <article className="analysis-card">
                            <h3>Imagen Original</h3>
                            <img
                                className="analysis-image"
                                src={`${API_BASE_URL}${selectedImage.url}`}
                                alt={selectedImage.filename}
                            />
                        </article>

                        <article className="analysis-card perturbation-card">
                            <h3>Mapa de Perturbación</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.perturbation_image_url}`}
                                    alt="Perturbation map"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Perturbación no generada
                                </div>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Imagen Adversarial</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.adversarial_image_url}`}
                                    alt="Adversarial"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Imagen adversarial no generada
                                </div>
                            )}
                        </article>
                    </div>

                    <div className="comparison-grid">
                        <article className="analysis-card">
                            <h3>Grad-CAM Original</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.original_gradcam_image_url}`}
                                    alt="Original Grad-CAM"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Heatmap original no generado
                                </div>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Grad-CAM Adversarial</h3>

                            {attackResult ? (
                                <img
                                    className="analysis-image"
                                    src={`${API_BASE_URL}${attackResult.adversarial_gradcam_image_url}`}
                                    alt="Adversarial Grad-CAM"
                                />
                            ) : (
                                <div className="image-placeholder">
                                    Heatmap adversarial no generado
                                </div>
                            )}
                        </article>

                        <article className="analysis-card prediction-main-card">
                            <h3>Predicción Original</h3>

                            {attackResult ? (
                                <>
                                    <p className="main-prediction">
                                        Clase predicha:{' '}
                                        <strong>
                                            {attackResult.original_prediction.prediction}
                                        </strong>
                                    </p>
                                    <p>
                                        Confianza:{' '}
                                        <strong>
                                            {attackResult.original_prediction.confidence}
                                        </strong>
                                    </p>
                                </>
                            ) : (
                                <p>No hay predicción original disponible.</p>
                            )}
                        </article>

                        <article className="analysis-card prediction-main-card">
                            <h3>Predicción Adversarial</h3>

                            {attackResult ? (
                                <>
                                    <p className="main-prediction danger-text">
                                        Clase predicha:{' '}
                                        <strong>
                                            {
                                                attackResult.adversarial_prediction
                                                    .prediction
                                            }
                                        </strong>
                                    </p>
                                    <p>
                                        Confianza:{' '}
                                        <strong>
                                            {
                                                attackResult.adversarial_prediction
                                                    .confidence
                                            }
                                        </strong>
                                    </p>
                                </>
                            ) : (
                                <p>No hay predicción adversarial disponible.</p>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Top-K Original</h3>

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
                                <p>No hay resultados Top-K disponibles.</p>
                            )}
                        </article>

                        <article className="analysis-card">
                            <h3>Top-K Adversarial</h3>

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
                                <p>No hay resultados Top-K disponibles.</p>
                            )}
                        </article>
                    </div>

                    <div className="comparison-detail-card">
                        <h3>Detalles del Ataque</h3>

                        {attackResult ? (
                            <div className="details-grid">
                                <p>
                                    <strong>Nombre del Ataque:</strong> {attackName}
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
                                    <strong>Perturbación L∞:</strong>{' '}
                                    {comparison.perturbation_linf}
                                </p>
                                <p>
                                    <strong>Perturbación L2:</strong>{' '}
                                    {comparison.perturbation_l2}
                                </p>
                                <p>
                                    <strong>Resultado del Ataque:</strong>{' '}
                                    {comparison.attack_success ? (
                                        <span className="status-danger">
                                            Exitoso
                                        </span>
                                    ) : (
                                        <span className="status-success">
                                            No exitoso
                                        </span>
                                    )}
                                </p>
                            </div>
                        ) : (
                            <p>Ejecuta un ataque para ver los detalles.</p>
                        )}
                    </div>

                    <div className="comparison-detail-card">
                        <h3>Análisis Comparativo</h3>

                        {attackResult ? (
                            <div className="details-grid">
                                <p>
                                    <strong>Predicción Cambiada:</strong>{' '}
                                    {comparison.prediction_changed ? (
                                        <span className="status-danger">
                                            Sí ({comparison.original_label} →{' '}
                                            {comparison.adversarial_label})
                                        </span>
                                    ) : (
                                        <span className="status-success">No</span>
                                    )}
                                </p>

                                <p>
                                    <strong>Diferencia de Confianza:</strong>{' '}
                                    {comparison.confidence_difference}
                                </p>

                                <p>
                                    <strong>Confianza Original:</strong>{' '}
                                    {comparison.original_confidence}
                                </p>

                                <p>
                                    <strong>Confianza Adversarial:</strong>{' '}
                                    {comparison.adversarial_confidence}
                                </p>

                                <p>
                                    <strong>Impacto del Ataque:</strong>{' '}
                                    {comparison.attack_success
                                        ? 'El ataque cambió la predicción del modelo.'
                                        : 'El ataque no cambió la predicción del modelo.'}
                                </p>
                            </div>
                        ) : (
                            <p>Ejecuta un ataque para ver el análisis comparativo.</p>
                        )}
                    </div>

                    <div className="analysis-actions comparison-bottom-actions">
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={() => navigate('/dashboard')}
                        >
                            Volver
                        </button>

                        <button
                            type="button"
                            className="primary-action"
                            onClick={handleRunAttack}
                            disabled={loading}
                        >
                            {loading ? 'Ejecutando...' : 'Ejecutar Nuevo Ataque'}
                        </button>
                    </div>
                </>
            )}
        </section>
    )
}

export default ComparisonView