import { useEffect, useState } from 'react'
import { evaluateDetectionDataset } from '../api/detectionEvaluation'

const ATTACK_NAMES = {
    random_noise: 'Random Noise',
    yolo_confidence: 'YOLO Confidence',
}

const SUPPORTED_DETECTION_ATTACKS = ['random_noise', 'yolo_confidence']

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

function formatDetections(detections) {
    if (!detections || detections.length === 0) {
        return '-'
    }

    return detections
        .map((detection) => `${detection.label} (${detection.confidence})`)
        .join(', ')
}

function DetectionEvaluationPanel({
    selectedDataset,
    selectedAttack,
    attackParameters,
}) {
    const [evaluation, setEvaluation] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const isSupportedAttack = SUPPORTED_DETECTION_ATTACKS.includes(selectedAttack)
    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack

    useEffect(() => {
        setEvaluation(null)
        setError('')
    }, [selectedDataset, selectedAttack, attackParameters])

    async function handleEvaluate() {
        if (!selectedDataset) {
            setError('Debes seleccionar un dataset.')
            return
        }

        if (!isSupportedAttack) {
            setError(
                'Para evaluar un modelo de detección debes seleccionar Random Noise o YOLO Confidence.',
            )
            return
        }

        try {
            setLoading(true)
            setError('')
            setEvaluation(null)

            const result = await evaluateDetectionDataset({
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

    return (
        <div className="evaluation-panel">
            {!selectedDataset ? (
                <div className="empty-state">
                    <h3>No hay dataset seleccionado</h3>
                    <p>
                        Vuelve al Dashboard y selecciona un dataset antes de ejecutar
                        la evaluación.
                    </p>
                </div>
            ) : (
                <>
                    <div className="evaluation-run-card">
                        <div>
                            <h3>Evaluación de Detección</h3>
                            <p>
                                Se evaluará el impacto de{' '}
                                <strong>{attackName}</strong> sobre todas las imágenes
                                del dataset usando el modelo de detección activo.
                            </p>

                            {!isSupportedAttack && (
                                <p className="muted-text">
                                    Ataque no compatible con detección. Selecciona
                                    Random Noise o YOLO Confidence en el Dashboard.
                                </p>
                            )}
                        </div>

                        <button
                            type="button"
                            className="primary-action"
                            onClick={handleEvaluate}
                            disabled={loading || !isSupportedAttack}
                        >
                            {loading ? 'Evaluando dataset...' : 'Ejecutar evaluación'}
                        </button>
                    </div>

                    {error && <p className="error">{error}</p>}

                    {evaluation && (
                        <>
                            <section className="metrics-section">
                                <h3>Métricas de Detección</h3>

                                <div className="metric-grid">
                                    <MetricCard
                                        title="Imágenes totales"
                                        value={summary.total_images}
                                    />

                                    <MetricCard
                                        title="Detecciones originales"
                                        value={summary.total_original_detections}
                                    />

                                    <MetricCard
                                        title="Detecciones adversarias"
                                        value={summary.total_adversarial_detections}
                                    />

                                    <MetricCard
                                        title="Diferencia de detecciones (original − adversaria)"
                                        value={summary.detections_difference}
                                        subtitle="Positiva: hay menos detecciones adversarias; no determina calidad sin ground truth."
                                    />

                                    <MetricCard
                                        title="Media detecciones original"
                                        value={summary.average_original_detections_per_image}
                                    />

                                    <MetricCard
                                        title="Media detecciones adversaria"
                                        value={summary.average_adversarial_detections_per_image}
                                    />

                                    <MetricCard
                                        title="Confianza media original"
                                        value={summary.average_original_confidence}
                                    />

                                    <MetricCard
                                        title="Confianza media adversaria"
                                        value={summary.average_adversarial_confidence}
                                    />

                                    <MetricCard
                                        title="Diferencia media de confianza (original − adversaria)"
                                        value={summary.average_confidence_difference}
                                        subtitle="Positiva: la confianza adversaria ha bajado."
                                    />

                                    <MetricCard
                                        title="Imágenes con menos detecciones"
                                        value={summary.images_with_reduced_detections}
                                    />

                                    <MetricCard
                                        title="Sin detecciones adversarias"
                                        value={summary.images_with_no_adversarial_detections}
                                    />

                                    <MetricCard
                                        title="Tasa de éxito legacy"
                                        value={summary.attack_success_rate}
                                        subtitle="Criterio legacy: reducción de detecciones o de confianza media."
                                    />

                                    <MetricCard
                                        title="Perturbación media L∞"
                                        value={summary.average_perturbation_linf}
                                    />

                                    <MetricCard
                                        title="Perturbación media L2"
                                        value={summary.average_perturbation_l2}
                                    />
                                </div>
                            </section>

                            <section className="charts-section">
                                <article className="analysis-card">
                                    <h3>Detecciones Totales</h3>

                                    <SimpleBar
                                        label="Originales"
                                        value={summary.total_original_detections}
                                        max={Math.max(
                                            summary.total_original_detections,
                                            summary.total_adversarial_detections,
                                            1,
                                        )}
                                    />

                                    <SimpleBar
                                        label="Adversarias"
                                        value={summary.total_adversarial_detections}
                                        max={Math.max(
                                            summary.total_original_detections,
                                            summary.total_adversarial_detections,
                                            1,
                                        )}
                                    />
                                </article>

                                <article className="analysis-card">
                                    <h3>Confianza Media</h3>

                                    <SimpleBar
                                        label="Original"
                                        value={summary.average_original_confidence}
                                        max={1}
                                    />

                                    <SimpleBar
                                        label="Adversaria"
                                        value={summary.average_adversarial_confidence}
                                        max={1}
                                    />
                                </article>

                                <article className="analysis-card">
                                    <h3>Impacto del Ataque</h3>

                                    <SimpleBar
                                        label="Imágenes afectadas"
                                        value={summary.images_with_reduced_detections}
                                        max={summary.total_images}
                                    />

                                    <SimpleBar
                                        label="Sin detecciones"
                                        value={summary.images_with_no_adversarial_detections}
                                        max={summary.total_images}
                                    />
                                </article>
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Resumen del Análisis</h3>

                                <p>
                                    El ataque redujo las detecciones en{' '}
                                    <strong>
                                        {summary.images_with_reduced_detections}
                                    </strong>{' '}
                                    de <strong>{summary.total_images}</strong> imágenes.
                                </p>

                                <p>
                                    El número total de detecciones pasó de{' '}
                                    <strong>{summary.total_original_detections}</strong>{' '}
                                    a{' '}
                                    <strong>
                                        {summary.total_adversarial_detections}
                                    </strong>
                                    .
                                </p>

                                <p>
                                    La confianza media cambió de{' '}
                                    <strong>{summary.average_original_confidence}</strong>{' '}
                                    a{' '}
                                    <strong>
                                        {summary.average_adversarial_confidence}
                                    </strong>
                                    , con una diferencia media de{' '}
                                    <strong>
                                        {summary.average_confidence_difference}
                                    </strong>
                                    {' '}
                                    (original − adversaria). Un valor positivo indica
                                    una caída de confianza.
                                </p>

                                <p>
                                    La diferencia de detecciones usa también original
                                    − adversaria. Un valor positivo indica menos
                                    detecciones adversarias, pero no permite concluir
                                    una mejora o degradación sin ground truth.
                                </p>
                            </section>

                            <section className="comparison-detail-card">
                                <h3>Resultados por Imagen</h3>

                                <div className="evaluation-table-wrapper">
                                    <table className="evaluation-table">
                                        <thead>
                                            <tr>
                                                <th>Imagen</th>
                                                <th>Det. Originales</th>
                                                <th>Det. Adv.</th>
                                                <th>Δ det. (orig. − adv.)</th>
                                                <th>Conf. Original</th>
                                                <th>Conf. Adv.</th>
                                                <th>Δ conf. (orig. − adv.)</th>
                                                <th>Éxito legacy</th>
                                                <th>Original</th>
                                                <th>Adversaria</th>
                                            </tr>
                                        </thead>

                                        <tbody>
                                            {evaluation.results.map((result) => (
                                                <tr key={result.image_id}>
                                                    <td>{result.filename}</td>
                                                    <td>
                                                        {
                                                            result.original_detection_count
                                                        }
                                                    </td>
                                                    <td>
                                                        {
                                                            result.adversarial_detection_count
                                                        }
                                                    </td>
                                                    <td>
                                                        {
                                                            result.detections_difference
                                                        }
                                                    </td>
                                                    <td>
                                                        {
                                                            result.average_original_confidence
                                                        }
                                                    </td>
                                                    <td>
                                                        {
                                                            result.average_adversarial_confidence
                                                        }
                                                    </td>
                                                    <td>
                                                        {
                                                            result.average_confidence_difference
                                                        }
                                                    </td>
                                                    <td>
                                                        {result.attack_success
                                                            ? 'Sí'
                                                            : 'No'}
                                                    </td>
                                                    <td className="detections-cell">
                                                        {formatDetections(
                                                            result.original_detections,
                                                        )}
                                                    </td>
                                                    <td className="detections-cell">
                                                        {formatDetections(
                                                            result.adversarial_detections,
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

export default DetectionEvaluationPanel
