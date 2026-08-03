import { useState } from 'react'
import { resolveApiUrl } from '../../api/client'
import { formatDecimal, formatInteger } from '../../utils/metricFormatting'

function formatDetections(detections) {
    if (!detections?.length) {
        return '—'
    }
    return detections
        .map((detection) => `${detection.label} (${formatDecimal(detection.confidence, 3)})`)
        .join(', ')
}

function DetectionOverlayImage({ datasetId, result, variant }) {
    const [dimensions, setDimensions] = useState(null)
    const [failed, setFailed] = useState(false)
    const isOriginal = variant === 'original'
    const detections = isOriginal
        ? result.original_detections
        : result.adversarial_detections
    const title = isOriginal ? 'Predicciones originales' : 'Predicciones adversariales'
    const imageUrl = resolveApiUrl(`/datasets/${datasetId}/images/${result.image_id}`)
    const fontSize = dimensions
        ? Math.max(12, Math.min(dimensions.width, dimensions.height) * 0.026)
        : 14
    const strokeWidth = dimensions
        ? Math.max(2, Math.min(dimensions.width, dimensions.height) * 0.005)
        : 2

    return (
        <figure className={`detection-preview tone-${variant}`}>
            <figcaption>
                <strong>{title}</strong>
                <span>{formatInteger(detections.length)} cajas</span>
            </figcaption>
            <div className="detection-preview-frame">
                {failed ? (
                    <div className="detection-preview-error">No se pudo cargar la imagen.</div>
                ) : (
                    <>
                        <img
                            src={imageUrl}
                            alt={`${result.filename}: ${title.toLocaleLowerCase('es')}`}
                            onLoad={(event) => {
                                setDimensions({
                                    width: event.currentTarget.naturalWidth,
                                    height: event.currentTarget.naturalHeight,
                                })
                            }}
                            onError={() => setFailed(true)}
                        />
                        {dimensions && (
                            <svg
                                className="detection-box-overlay"
                                viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                                aria-label={`${detections.length} detecciones ${isOriginal ? 'originales' : 'adversariales'}`}
                                role="img"
                            >
                                {detections.map((detection, index) => {
                                    const { x1, y1, x2, y2 } = detection.bbox
                                    const label = `${detection.label} ${formatDecimal(detection.confidence, 2)}`
                                    const labelWidth = Math.min(
                                        x2 - x1,
                                        Math.max(fontSize * 4, label.length * fontSize * 0.58),
                                    )
                                    const labelY = Math.max(0, y1 - fontSize * 1.4)

                                    return (
                                        <g key={`${detection.label}-${index}`}>
                                            <rect
                                                className="detection-overlay-box"
                                                x={x1}
                                                y={y1}
                                                width={x2 - x1}
                                                height={y2 - y1}
                                                strokeWidth={strokeWidth}
                                            />
                                            <rect
                                                className="detection-overlay-label-bg"
                                                x={x1}
                                                y={labelY}
                                                width={labelWidth}
                                                height={fontSize * 1.4}
                                            />
                                            <text
                                                className="detection-overlay-label"
                                                x={x1 + fontSize * 0.25}
                                                y={labelY + fontSize}
                                                fontSize={fontSize}
                                            >
                                                {label}
                                            </text>
                                        </g>
                                    )
                                })}
                            </svg>
                        )}
                    </>
                )}
            </div>
        </figure>
    )
}

function DetectionImageResults({ evaluation }) {
    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-surface detection-table-surface">
                <div className="evaluation-section-heading">
                    <div>
                        <span className="evaluation-eyebrow">Salida interactiva legacy</span>
                        <h3>Predicciones por imagen</h3>
                        <p>Estas detecciones usan el filtro público estricto de confianza &gt; 0.25 y no sustituyen las métricas contra ground truth.</p>
                    </div>
                    <strong>{formatInteger(evaluation.results.length)} imágenes</strong>
                </div>

                <div className="detection-image-results-grid">
                    {evaluation.results.map((result) => (
                        <article className="detection-image-result-card" key={result.image_id}>
                            <header>
                                <div>
                                    <strong>{result.filename}</strong>
                                    <small>{result.image_id}</small>
                                </div>
                                <span className={`detection-legacy-badge ${result.attack_success ? 'success' : 'neutral'}`}>
                                    {result.attack_success ? 'Éxito legacy' : 'Sin éxito legacy'}
                                </span>
                            </header>

                            <div className="detection-preview-grid">
                                <DetectionOverlayImage
                                    datasetId={evaluation.dataset_id}
                                    result={result}
                                    variant="original"
                                />
                                <DetectionOverlayImage
                                    datasetId={evaluation.dataset_id}
                                    result={result}
                                    variant="adversarial"
                                />
                            </div>

                            <p className="detection-preview-note">
                                Las cajas adversariales se proyectan sobre la imagen fuente
                                como referencia geométrica; la perturbación no se vuelve a
                                generar ni se ejecuta otra inferencia.
                            </p>

                            <dl className="detection-image-result-metrics">
                                <div><dt>Det. original</dt><dd>{formatInteger(result.original_detection_count)}</dd></div>
                                <div><dt>Det. adversaria</dt><dd>{formatInteger(result.adversarial_detection_count)}</dd></div>
                                <div><dt>Δ detecciones</dt><dd>{formatInteger(result.detections_difference)}</dd></div>
                                <div><dt>Conf. original</dt><dd>{formatDecimal(result.average_original_confidence)}</dd></div>
                                <div><dt>Conf. adversaria</dt><dd>{formatDecimal(result.average_adversarial_confidence)}</dd></div>
                                <div><dt>Δ confianza</dt><dd>{formatDecimal(result.average_confidence_difference)}</dd></div>
                            </dl>

                            <details className="detection-prediction-list">
                                <summary>Ver predicciones en texto</summary>
                                <dl>
                                    <div><dt>Originales</dt><dd>{formatDetections(result.original_detections)}</dd></div>
                                    <div><dt>Adversariales</dt><dd>{formatDetections(result.adversarial_detections)}</dd></div>
                                </dl>
                            </details>
                        </article>
                    ))}
                </div>
            </section>
        </div>
    )
}

export default DetectionImageResults
