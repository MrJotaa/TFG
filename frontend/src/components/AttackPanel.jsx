import { useEffect, useState } from 'react'
import { applyAttack } from '../api/attack'

function AttackPanel({ selectedDataset, selectedImage }) {
    const [attackType, setAttackType] = useState('fgsm')
    const [epsilon, setEpsilon] = useState(0.03)
    const [attackResult, setAttackResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        setAttackResult(null)
        setError('')
    }, [selectedDataset, selectedImage])

    async function handleApplyAttack() {
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
                attackType,
                epsilon: Number(epsilon),
            })

            setAttackResult(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <h2>Ataque adversario</h2>

            {!selectedImage ? (
                <p>Selecciona una imagen para aplicar un ataque adversario.</p>
            ) : (
                <p>
                    Imagen seleccionada: <strong>{selectedImage.filename}</strong>
                </p>
            )}

            {error && <p className="error">{error}</p>}

            <label>
                Tipo de ataque
                <select
                    value={attackType}
                    onChange={(event) => setAttackType(event.target.value)}
                >
                    <option value="fgsm">FGSM</option>
                    <option value="random_noise">Random Noise</option>
                </select>
            </label>

            <label>
                Epsilon: {epsilon}
                <input
                    type="range"
                    min="0"
                    max="0.3"
                    step="0.01"
                    value={epsilon}
                    onChange={(event) => setEpsilon(event.target.value)}
                />
            </label>

            <button
                type="button"
                onClick={handleApplyAttack}
                disabled={!selectedDataset || !selectedImage || loading}
            >
                {loading ? 'Aplicando ataque...' : 'Aplicar ataque'}
            </button>

            {attackResult && selectedImage && (
                <div>
                    <h3>Resultado del ataque</h3>

                    <div>
                        <h4>Imágenes</h4>

                        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                            <div>
                                <p>Imagen original</p>
                                <img
                                    src={`http://127.0.0.1:8000${selectedImage.url}`}
                                    alt="Imagen original"
                                    width={220}
                                />
                            </div>

                            <div>
                                <p>Perturbación</p>
                                <img
                                    src={`http://127.0.0.1:8000${attackResult.perturbation_image_url}`}
                                    alt="Perturbación"
                                    width={220}
                                />
                            </div>

                            <div>
                                <p>Imagen adversaria</p>
                                <img
                                    src={`http://127.0.0.1:8000${attackResult.adversarial_image_url}`}
                                    alt="Imagen adversaria"
                                    width={220}
                                />
                            </div>
                        </div>
                    </div>

                    <div>
                        <h4>Predicción original</h4>
                        <p>
                            Clase: <strong>{attackResult.original_prediction.prediction}</strong>
                        </p>
                        <p>
                            Confianza:{' '}
                            <strong>{attackResult.original_prediction.confidence}</strong>
                        </p>
                    </div>

                    <div>
                        <h4>Predicción adversaria</h4>
                        <p>
                            Clase:{' '}
                            <strong>{attackResult.adversarial_prediction.prediction}</strong>
                        </p>
                        <p>
                            Confianza:{' '}
                            <strong>{attackResult.adversarial_prediction.confidence}</strong>
                        </p>
                    </div>

                    <div>
                        <h4>Comparación</h4>
                        <p>
                            Predicción cambiada:{' '}
                            <strong>
                                {attackResult.comparison.prediction_changed ? 'Sí' : 'No'}
                            </strong>
                        </p>
                        <p>
                            Ataque exitoso:{' '}
                            <strong>
                                {attackResult.comparison.attack_success ? 'Sí' : 'No'}
                            </strong>
                        </p>
                        <p>
                            Cambio de confianza:{' '}
                            <strong>{attackResult.comparison.confidence_difference}</strong>
                        </p>
                        <p>
                            Perturbación L∞:{' '}
                            <strong>{attackResult.comparison.perturbation_linf}</strong>
                        </p>
                        <p>
                            Perturbación L2:{' '}
                            <strong>{attackResult.comparison.perturbation_l2}</strong>
                        </p>
                    </div>

                    <div>
                        <h4>Grad-CAM</h4>

                        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                            <div>
                                <p>Grad-CAM original</p>
                                <img
                                    src={`http://127.0.0.1:8000${attackResult.original_gradcam_image_url}`}
                                    alt="Grad-CAM original"
                                    width={220}
                                />
                            </div>

                            <div>
                                <p>Grad-CAM adversario</p>
                                <img
                                    src={`http://127.0.0.1:8000${attackResult.adversarial_gradcam_image_url}`}
                                    alt="Grad-CAM adversario"
                                    width={220}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default AttackPanel