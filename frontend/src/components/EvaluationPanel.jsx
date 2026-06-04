import { useEffect, useState } from 'react'
import { evaluateDataset } from '../api/evaluation'

const API_BASE_URL = 'http://127.0.0.1:8000'

const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
}

function MetricCard({ title, value, subtitle }) {
    return (
        <article className="metric-card">
            <span>{title}</span>
            <strong>{value}</strong>
            {subtitle && <small>{subtitle}</small>}
        </article>
    )
}

function SimpleBar({ label, value, max = 1 }) {
    const safeMax = max || 1
    const percentage = Math.max(0, Math.min(100, (value / safeMax) * 100))

    return (
        <div className="simple-bar-row">
            <div className="simple-bar-label">
                <span>{label}</span>
                <strong>{value}</strong>
            </div>

            <div className="simple-bar-track">
                <div
                    className="simple-bar-fill"
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    )
}

function EvaluationPanel({
    selectedDataset,
    selectedAttack = 'fgsm',
    attackParameters = {
        epsilon: 0.03,
        alpha: 0.01,
        steps: 10,
    },
}) {
    const [evaluation, setEvaluation] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        setEvaluation(null)
        setError('')
    }, [selectedDataset, selectedAttack, attackParameters])

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
                attackType: selectedAttack,
                epsilon: attackParameters.epsilon,
                alpha: attackParameters.alpha,
                steps: attackParameters.steps,
            })

            setEvaluation(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const summary = evaluation?.summary
    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack

    return (
        <div className="evaluation-panel">
            {!selectedDataset ? (
                <div className="empty-state">
                    <h3>No hay dataset seleccionado</h3>
                    <p>
                        Vuelve al dashboard y selecciona un dataset antes de ejecutar
                        la evaluación.
                    </p>
                </div>
            ) : (
                <>
                    <div className="evaluation-run-card">
                        <div>
                            <h3>Ejecutar la evaluación del dataset</h3>
                            <p>
                                Se aplicará <strong>{attackName}</strong> sobre todas
                                las imágenes del dataset seleccionado.
                            </p>
                        </div>

                        <button
                            type="button"
                            className="primary-action"
                            onClick={handleEvaluate}
                            disabled={loading}
                        >
                            {loading ? 'Evaluando dataset...' : 'Ejecutar Evaluación'}
                        </button>
                    </div>

                    {error && <p className="error">{error}</p>}

                    {evaluation && (
                        <>
                            <section className="metrics-section">
                                <h3>Métricas de Evaluación</h3>

                                <div className="metric-grid">
                                    <MetricCard
                                        title="Imágenes Totales"
                                        value={summary.total_images}
                                    />

                                    <MetricCard
                                        title="Predicciones Cambiadas"
                                        value={summary.changed_predictions}
                                        subtitle={`${summary.unchanged_predictions} sin cambiar`}
                                    />

                                    <MetricCard
                                        title="Tasa de Éxito del Ataque"
                                        value={summary.attack_success_rate}
                                    />

                                    <MetricCard
                                        title="Media: Confianza Δ"
                                        value={summary.average_confidence_difference}
                                    />

                                    <MetricCard
                                        title="Media: Confianza Original"
                                        value={summary.average_original_confidence}
                                    />

                                    <MetricCard
                                        title="Media: Confianza Adversarial"
                                        value={summary.average_adversarial_confidence}
                                    />

                                    <MetricCard
                                        title="Media: PerturbaciónL∞"
                                        value={summary.average_perturbation_linf}
                                    />

                                    <MetricCard
                                        title="Media: Perturbación L2"
                                        value={summary.average_perturbation_l2}
                                    />
                                </div>
                            </section>

                            {summary.labeled_images > 0 && (
                                <section className="metrics-section">
                                    <h3>Métricas de Ground Truth</h3>

                                    <div className="metric-grid">
                                        <MetricCard
                                            title="Imágenes Etiquetadas"
                                            value={summary.labeled_images}
                                        />

                                        <MetricCard
                                            title="Accuracy Original"
                                            value={summary.original_accuracy}
                                        />

                                        <MetricCard
                                            title="Accuracy Adversarial"
                                            value={summary.adversarial_accuracy}
                                        />

                                        <MetricCard
                                            title="Accuracy Drop"
                                            value={summary.accuracy_drop}
                                        />

                                        <MetricCard
                                            title="Correctas antes / Incorrectas después"
                                            value={summary.correct_before_wrong_after}
                                        />
                                    </div>
                                </section>
                            )}

                            {summary.labeled_images === 0 && (
                                <div className="info-card">
                                    Este dataset no tiene etiquetas. Se muestran métricas
                                    de cambio de predicción y confianza, pero no métricas
                                    de accuracy.
                                </div>
                            )}

                            <section className="charts-section">
                                <article className="analysis-card">
                                    <h3>Modificado vs No Modificado</h3>

                                    <SimpleBar
                                        label="Changed"
                                        value={summary.changed_predictions}
                                        max={summary.total_images}
                                    />

                                    <SimpleBar
                                        label="Unchanged"
                                        value={summary.unchanged_predictions}
                                        max={summary.total_images}
                                    />
                                </article>

                                <article className="analysis-card">
                                    <h3>Confianza Antes vs Despues</h3>

                                    <SimpleBar
                                        label="Original"
                                        value={summary.average_original_confidence}
                                        max={1}
                                    />

                                    <SimpleBar
                                        label="Adversarial"
                                        value={summary.average_adversarial_confidence}
                                        max={1}
                                    />
                                </article>

                                {summary.labeled_images > 0 && (
                                    <article className="analysis-card">
                                        <h3>Accuracy Antes vs Despues</h3>

                                        <SimpleBar
                                            label="Accuracy original"
                                            value={summary.original_accuracy}
                                            max={1}
                                        />

                                        <SimpleBar
                                            label="Accuracy adversarial"
                                            value={summary.adversarial_accuracy}
                                            max={1}
                                        />
                                    </article>
                                )}
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Resumen del Análisis</h3>

                                <p>
                                    El ataque cambió{' '}
                                    <strong>{summary.changed_predictions}</strong> de{' '}
                                    <strong>{summary.total_images}</strong> predicciones,
                                    con una tasa de éxito en los ataques de{' '}
                                    <strong>{summary.attack_success_rate}</strong>.
                                </p>

                                <p>
                                    La confianza promedio cambió en{' '}
                                    <strong>
                                        {summary.average_confidence_difference}
                                    </strong>
                                    .
                                </p>

                                {summary.labeled_images > 0 && (
                                    <p>
                                        Usando etiquetas ground-truth, la accuracy ha pasado de{' '}
                                        <strong>{summary.original_accuracy}</strong> a{' '}
                                        <strong>{summary.adversarial_accuracy}</strong>,
                                        produciendo una caida de accuracy de{' '}
                                        <strong>{summary.accuracy_drop}</strong>.
                                    </p>
                                )}
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Resultados por Imagen</h3>

                                <div className="evaluation-table-wrapper">
                                    <table className="evaluation-table">
                                        <thead>
                                            <tr>
                                                <th>Imagen</th>
                                                <th>Ground Truth</th>
                                                <th>Original</th>
                                                <th>Adv.</th>
                                                <th>Conf. Δ</th>
                                                <th> Cambio</th>
                                                <th>Resultado del Ataque</th>
                                                <th>Adversarial</th>
                                                <th>Perturbación</th>
                                            </tr>
                                        </thead>

                                        <tbody>
                                            {evaluation.results.map((result) => (
                                                <tr key={result.image_id}>
                                                    <td>{result.filename}</td>
                                                    <td>
                                                        {result.ground_truth_label ||
                                                            '-'}
                                                    </td>
                                                    <td>{result.original_label}</td>
                                                    <td>{result.adversarial_label}</td>
                                                    <td>
                                                        {result.confidence_difference}
                                                    </td>
                                                    <td>
                                                        {result.prediction_changed
                                                            ? 'Sí'
                                                            : 'No'}
                                                    </td>
                                                    <td>
                                                        {result.attack_success
                                                            ? 'Exitoso'
                                                            : 'No exitoso'}
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
                                                                alt={`Perturbation ${result.filename}`}
                                                                width={80}
                                                            />
                                                        )}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        </>
                    )}
                </>
            )}
        </div>
    )
}

export default EvaluationPanel