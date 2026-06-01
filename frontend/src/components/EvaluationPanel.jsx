import { useEffect, useState } from 'react'
import { evaluateDataset } from '../api/evaluation'

const API_BASE_URL = 'http://127.0.0.1:8000'

function EvaluationPanel({ selectedDataset }) {
    const [attackType, setAttackType] = useState('fgsm')
    const [epsilon, setEpsilon] = useState(0.03)
    const [evaluation, setEvaluation] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [alpha, setAlpha] = useState(0.01)
    const [steps, setSteps] = useState(10)

    useEffect(() => {
        setEvaluation(null)
        setError('')
    }, [selectedDataset])

    async function handleEvaluate() {
        if (!selectedDataset) {
            setError('Debes seleccionar un dataset.')
            return
        }

        try {
            setLoading(true)
            setError('')
            setEvaluation(null)

            const result = await evaluateDataset({
                datasetId: selectedDataset.id,
                attackType,
                epsilon,
                alpha,
                steps,
            })

            setEvaluation(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <h2>Evaluación del dataset</h2>

            {!selectedDataset ? (
                <p>Selecciona un dataset para ejecutar una evaluación.</p>
            ) : (
                <p>
                    Dataset seleccionado: <strong>{selectedDataset.name}</strong>
                </p>
            )}

            {error && <p className="error">{error}</p>}

            <div>
                <label>
                    Ataque
                    <select
                        value={attackType}
                        onChange={(event) => setAttackType(event.target.value)}
                    >
                        <option value="fgsm">FGSM</option>
                        <option value="random_noise">Random Noise</option>
                        <option value="pgd">PGD</option>
                    </select>
                </label>
            </div>

            <div>
                <label>
                    Epsilon: {epsilon}
                    <input
                        type="range"
                        min="0"
                        max="0.1"
                        step="0.01"
                        value={epsilon}
                        onChange={(event) => setEpsilon(event.target.value)}
                    />
                    {attackType === 'pgd' && (
                        <div>
                            <label>
                                Alpha: {alpha}
                                <input
                                    type="range"
                                    min="0.001"
                                    max="0.05"
                                    step="0.001"
                                    value={alpha}
                                    onChange={(event) => setAlpha(event.target.value)}
                                />
                            </label>

                            <label>
                                Steps: {steps}
                                <input
                                    type="number"
                                    min="1"
                                    max="100"
                                    value={steps}
                                    onChange={(event) => setSteps(event.target.value)}
                                />
                            </label>
                        </div>
                    )}
                </label>
            </div>

            <button
                type="button"
                onClick={handleEvaluate}
                disabled={!selectedDataset || loading}
            >
                {loading ? 'Evaluando dataset...' : 'Evaluar dataset'}
            </button>

            {evaluation && (
                <div>
                    <h3>Resumen de evaluación</h3>

                    <div>
                        <p>
                            Imágenes evaluadas:{' '}
                            <strong>{evaluation.summary.total_images}</strong>
                        </p>
                        <p>
                            Predicciones cambiadas:{' '}
                            <strong>{evaluation.summary.changed_predictions}</strong>
                        </p>
                        <p>
                            Predicciones sin cambio:{' '}
                            <strong>{evaluation.summary.unchanged_predictions}</strong>
                        </p>
                        <p>
                            Tasa de éxito del ataque:{' '}
                            <strong>{evaluation.summary.attack_success_rate}</strong>
                        </p>
                        <p>
                            Confianza media original:{' '}
                            <strong>
                                {evaluation.summary.average_original_confidence}
                            </strong>
                        </p>
                        <p>
                            Confianza media adversaria:{' '}
                            <strong>
                                {evaluation.summary.average_adversarial_confidence}
                            </strong>
                        </p>
                        <p>
                            Diferencia media de confianza:{' '}
                            <strong>
                                {evaluation.summary.average_confidence_difference}
                            </strong>
                        </p>
                    </div>

                    <h3>Resultados por imagen</h3>

                    <div style={{ overflowX: 'auto' }}>
                        <table border="1" cellPadding="6">
                            <thead>
                                <tr>
                                    <th>Imagen</th>
                                    <th>Original</th>
                                    <th>Conf. original</th>
                                    <th>Adversaria</th>
                                    <th>Conf. adversaria</th>
                                    <th>Cambió</th>
                                    <th>Adversarial</th>
                                    <th>Perturbación</th>
                                </tr>
                            </thead>
                            <tbody>
                                {evaluation.results.map((result) => (
                                    <tr key={result.image_id}>
                                        <td>{result.filename}</td>
                                        <td>{result.original_label}</td>
                                        <td>{result.original_confidence}</td>
                                        <td>{result.adversarial_label}</td>
                                        <td>{result.adversarial_confidence}</td>
                                        <td>
                                            {result.prediction_changed ? 'Sí' : 'No'}
                                        </td>
                                        <td>
                                            {result.adversarial_image_url && (
                                                <img
                                                    src={`${API_BASE_URL}${result.adversarial_image_url}`}
                                                    alt={`Adversarial ${result.filename}`}
                                                    width={80}
                                                />
                                            )}
                                        </td>
                                        <td>
                                            {result.perturbation_image_url && (
                                                <img
                                                    src={`${API_BASE_URL}${result.perturbation_image_url}`}
                                                    alt={`Perturbación ${result.filename}`}
                                                    width={80}
                                                />
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}

export default EvaluationPanel