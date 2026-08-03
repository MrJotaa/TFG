import { useState } from 'react'
import { resolveApiUrl } from '../../api/client'
import {
    getDetectionEigenCamEntry,
    getDetectionEigenCamKey,
} from '../../utils/detectionEvaluationEigenCamState'
import { formatDecimal, formatInteger } from '../../utils/metricFormatting'

const PAGE_SIZE = 6

function formatDetections(detections) {
    if (!detections?.length) {
        return '—'
    }
    return detections
        .map((detection) => `${detection.label} (${formatDecimal(detection.confidence, 3)})`)
        .join(', ')
}

function DetectionOverlayImage({ imagePath, result, variant }) {
    const [dimensions, setDimensions] = useState(null)
    const [failed, setFailed] = useState(false)
    const isOriginal = variant === 'original'
    const detections = isOriginal
        ? result.original_detections
        : result.adversarial_detections
    const title = isOriginal ? 'Original' : 'Adversaria'
    const imageUrl = resolveApiUrl(imagePath)
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
                {!imageUrl || failed ? (
                    <div className="detection-preview-error">
                        Imagen temporal no disponible.
                    </div>
                ) : (
                    <>
                        <img
                            src={imageUrl}
                            alt={`${result.filename}: imagen ${title.toLocaleLowerCase('es')}`}
                            loading="lazy"
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

function EigenCamPreview({ label, path, filename }) {
    const imageUrl = resolveApiUrl(path)

    return (
        <figure className="detection-eigencam-preview">
            <figcaption>{label}</figcaption>
            <img
                src={imageUrl}
                alt={`${label} de ${filename}`}
                loading="lazy"
            />
        </figure>
    )
}

function DetectionImageCard({
    evaluation,
    result,
    eigenCamEntry,
    onGenerateEigenCam,
}) {
    const eigenCamLoading = eigenCamEntry.status === 'loading'
    const eigenCamReady = eigenCamEntry.status === 'success'
    const eigenCamError = eigenCamEntry.status === 'error'
        ? eigenCamEntry.error
        : ''
    const originalImagePath = (
        `/datasets/${evaluation.dataset_id}/images/${result.image_id}`
    )

    return (
        <article className="detection-image-result-card">
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
                    imagePath={originalImagePath}
                    result={result}
                    variant="original"
                />
                <DetectionOverlayImage
                    imagePath={result.adversarial_image_url}
                    result={result}
                    variant="adversarial"
                />
            </div>

            <p className="detection-preview-note">
                Las dos vistas conservan la geometría fuente. La imagen adversaria
                es el resultado real del ataque, no una proyección sobre la original.
            </p>

            <dl className="detection-image-result-metrics">
                <div><dt>Det. original</dt><dd>{formatInteger(result.original_detection_count)}</dd></div>
                <div><dt>Det. adversaria</dt><dd>{formatInteger(result.adversarial_detection_count)}</dd></div>
                <div><dt>Δ detecciones</dt><dd>{formatInteger(result.detections_difference)}</dd></div>
                <div><dt>Conf. original</dt><dd>{formatDecimal(result.average_original_confidence)}</dd></div>
                <div><dt>Conf. adversaria</dt><dd>{formatDecimal(result.average_adversarial_confidence)}</dd></div>
                <div><dt>Δ confianza</dt><dd>{formatDecimal(result.average_confidence_difference)}</dd></div>
            </dl>

            <section className="detection-eigencam-section">
                <div className="detection-eigencam-action">
                    <div>
                        <strong>Explicabilidad EigenCAM</strong>
                        <span>
                            Genera los mapas original y adversario solo para esta imagen.
                        </span>
                    </div>
                    <button
                        type="button"
                        className="evaluation-gradcam-button"
                        disabled={eigenCamLoading || eigenCamReady}
                        onClick={() => onGenerateEigenCam(
                            evaluation.result_id,
                            result.image_id,
                        )}
                    >
                        {eigenCamLoading
                            ? 'Generando EigenCAM…'
                            : eigenCamReady
                              ? 'EigenCAM generado'
                              : eigenCamError
                                ? 'Reintentar EigenCAM'
                                : 'Generar EigenCAM'}
                    </button>
                </div>

                {eigenCamLoading && (
                    <p className="evaluation-gradcam-status" role="status">
                        Calculando EigenCAM sin repetir el ataque ni las detecciones.
                    </p>
                )}

                {eigenCamError && (
                    <div className="evaluation-gradcam-error" role="alert">
                        {eigenCamError}
                    </div>
                )}

                {eigenCamReady && (
                    <div className="detection-eigencam-grid">
                        <EigenCamPreview
                            label="EigenCAM original"
                            path={eigenCamEntry.data.original_eigencam_image_url}
                            filename={result.filename}
                        />
                        <EigenCamPreview
                            label="EigenCAM adversario"
                            path={eigenCamEntry.data.adversarial_eigencam_image_url}
                            filename={result.filename}
                        />
                    </div>
                )}
            </section>

            <details className="detection-prediction-list">
                <summary>Ver predicciones en texto</summary>
                <dl>
                    <div><dt>Originales</dt><dd>{formatDetections(result.original_detections)}</dd></div>
                    <div><dt>Adversariales</dt><dd>{formatDetections(result.adversarial_detections)}</dd></div>
                </dl>
            </details>
        </article>
    )
}

function DetectionImageResults({
    evaluation,
    eigenCamStates,
    onGenerateEigenCam,
}) {
    const [page, setPage] = useState(1)
    const totalPages = Math.max(
        1,
        Math.ceil(evaluation.results.length / PAGE_SIZE),
    )
    const safePage = Math.min(page, totalPages)
    const pageStart = (safePage - 1) * PAGE_SIZE
    const pageResults = evaluation.results.slice(
        pageStart,
        pageStart + PAGE_SIZE,
    )

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-surface detection-table-surface">
                <div className="evaluation-section-heading">
                    <div>
                        <span className="evaluation-eyebrow">Exploración visual</span>
                        <h3>Predicciones por imagen</h3>
                        <p>
                            Compara los píxeles y cajas originales y adversariales, y
                            solicita EigenCAM únicamente para los casos relevantes. Las
                            cajas mantienen el filtro público legacy de confianza &gt; 0.25.
                        </p>
                    </div>
                    <strong>{formatInteger(evaluation.results.length)} imágenes</strong>
                </div>

                <div className="detection-image-results-grid">
                    {pageResults.map((result) => (
                        <DetectionImageCard
                            key={result.image_id}
                            evaluation={evaluation}
                            result={result}
                            eigenCamEntry={getDetectionEigenCamEntry(
                                eigenCamStates,
                                getDetectionEigenCamKey(
                                    evaluation.result_id,
                                    result.image_id,
                                ),
                            )}
                            onGenerateEigenCam={onGenerateEigenCam}
                        />
                    ))}
                </div>

                {totalPages > 1 && (
                    <nav className="evaluation-pagination" aria-label="Paginación de imágenes de detección">
                        <button
                            type="button"
                            disabled={safePage === 1}
                            onClick={() => setPage((current) => Math.max(1, current - 1))}
                        >
                            Anterior
                        </button>
                        <span aria-live="polite">
                            Página <strong aria-current="page">{safePage}</strong> de{' '}
                            <strong>{totalPages}</strong>
                        </span>
                        <button
                            type="button"
                            disabled={safePage === totalPages}
                            onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                        >
                            Siguiente
                        </button>
                    </nav>
                )}
            </section>
        </div>
    )
}

export default DetectionImageResults
