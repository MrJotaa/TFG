import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { resolveApiUrl } from '../../api/client'
import {
    getDetectionEigenCamButtonLabel,
    getDetectionEigenCamEntry,
    getDetectionEigenCamKey,
} from '../../utils/detectionEvaluationEigenCamState'
import {
    clampEvaluationPreviewPan,
    clampEvaluationPreviewZoom,
    getEvaluationPreviewPanBounds,
    stepEvaluationPreviewZoom,
} from '../../utils/evaluationImagePreview'
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

function ExpandIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8.5 4H4v4.5M15.5 4H20v4.5M20 15.5V20h-4.5M8.5 20H4v-4.5" />
        </svg>
    )
}

function CloseIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m6 6 12 12M18 6 6 18" />
        </svg>
    )
}

function ZoomOutIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 12h14" />
        </svg>
    )
}

function ZoomInIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 5v14M5 12h14" />
        </svg>
    )
}

function DetectionBoxesOverlay({ detections, dimensions }) {
    if (!dimensions || !detections?.length) return null

    const fontSize = Math.max(
        12,
        Math.min(dimensions.width, dimensions.height) * 0.026,
    )
    const strokeWidth = Math.max(
        2,
        Math.min(dimensions.width, dimensions.height) * 0.005,
    )

    return (
        <svg
            className="detection-box-overlay"
            viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
            aria-label={`${detections.length} detecciones`}
            >
            <title>{`${detections.length} detecciones`}</title>
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
    )
}

function DetectionOverlayImage({ imagePath, result, variant, onExpand }) {
    const [dimensions, setDimensions] = useState(null)
    const [failed, setFailed] = useState(false)
    const isOriginal = variant === 'original'
    const detections = isOriginal
        ? result.original_detections
        : result.adversarial_detections
    const title = isOriginal ? 'Original' : 'Tras el ataque'
    const imageUrl = resolveApiUrl(imagePath)
    const alt = `${result.filename}: imagen ${title.toLocaleLowerCase('es')}`

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
                            alt={alt}
                            loading="lazy"
                            onLoad={(event) => {
                                setDimensions({
                                    width: event.currentTarget.naturalWidth,
                                    height: event.currentTarget.naturalHeight,
                                })
                            }}
                            onError={() => setFailed(true)}
                        />
                        <DetectionBoxesOverlay
                            detections={detections}
                            dimensions={dimensions}
                        />
                        <button
                            type="button"
                            className="evaluation-image-expand detection-image-expand"
                            aria-label={`Ampliar ${title.toLocaleLowerCase('es')}`}
                            title={`Ampliar ${title.toLocaleLowerCase('es')}`}
                            onClick={() => onExpand({
                                label: title,
                                url: imageUrl,
                                alt,
                                detections,
                                variant,
                            })}
                        >
                            <ExpandIcon />
                        </button>
                    </>
                )}
            </div>
        </figure>
    )
}

function EigenCamPreview({ label, path, filename, variant, onExpand }) {
    const imageUrl = resolveApiUrl(path)
    const alt = `${label} de ${filename}`

    return (
        <figure className="detection-eigencam-preview">
            <figcaption>{label}</figcaption>
            <div className="detection-eigencam-visual">
                <img
                    src={imageUrl}
                    alt={alt}
                    loading="lazy"
                />
                <button
                    type="button"
                    className="evaluation-image-expand detection-image-expand"
                    aria-label={`Ampliar ${label.toLocaleLowerCase('es')}`}
                    title={`Ampliar ${label.toLocaleLowerCase('es')}`}
                    onClick={() => onExpand({
                        label,
                        url: imageUrl,
                        alt,
                        detections: null,
                        variant,
                    })}
                >
                    <ExpandIcon />
                </button>
            </div>
        </figure>
    )
}

function DetectionImageLightbox({ preview, onClose }) {
    const [zoom, setZoom] = useState(1)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [isDragging, setIsDragging] = useState(false)
    const [dimensions, setDimensions] = useState(null)
    const viewportRef = useRef(null)
    const canvasRef = useRef(null)
    const dragRef = useRef(null)

    useEffect(() => {
        const previousOverflow = document.body.style.overflow
        document.body.style.overflow = 'hidden'

        function handleKeyDown(event) {
            if (event.key === 'Escape') onClose()
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => {
            document.body.style.overflow = previousOverflow
            window.removeEventListener('keydown', handleKeyDown)
        }
    }, [onClose])

    function getPanBounds(nextZoom = zoom) {
        const viewport = viewportRef.current
        const canvas = canvasRef.current

        if (!viewport || !canvas) return { x: 0, y: 0 }

        return getEvaluationPreviewPanBounds({
            viewportWidth: viewport.clientWidth,
            viewportHeight: viewport.clientHeight,
            imageWidth: canvas.offsetWidth,
            imageHeight: canvas.offsetHeight,
            zoom: nextZoom,
        })
    }

    function changeZoom(direction) {
        const nextZoom = stepEvaluationPreviewZoom(zoom, direction)
        setZoom(nextZoom)
        setPan((current) => clampEvaluationPreviewPan(current, getPanBounds(nextZoom)))
    }

    function resetView() {
        setZoom(1)
        setPan({ x: 0, y: 0 })
        setIsDragging(false)
        dragRef.current = null
    }

    function handlePointerDown(event) {
        if (zoom <= 1) return

        event.currentTarget.setPointerCapture(event.pointerId)
        dragRef.current = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            panX: pan.x,
            panY: pan.y,
        }
        setIsDragging(true)
    }

    function handlePointerMove(event) {
        const drag = dragRef.current
        if (!drag || drag.pointerId !== event.pointerId || zoom <= 1) return

        setPan(clampEvaluationPreviewPan({
            x: drag.panX + event.clientX - drag.startX,
            y: drag.panY + event.clientY - drag.startY,
        }, getPanBounds()))
    }

    function finishPointerDrag(event) {
        const drag = dragRef.current
        if (!drag || drag.pointerId !== event.pointerId) return

        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId)
        }
        dragRef.current = null
        setIsDragging(false)
    }

    const safeZoom = clampEvaluationPreviewZoom(zoom)
    const modal = (
        <div
            className="image-lightbox-backdrop evaluation-image-lightbox-backdrop"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget) onClose()
            }}
        >
            <dialog
                open
                className="image-lightbox evaluation-image-lightbox detection-evaluation-image-lightbox"
                aria-modal="true"
                aria-label={`Vista ampliada: ${preview.label}`}
            >
                <header>
                    <div>
                        <span className="evaluation-eyebrow">Vista ampliada</span>
                        <strong>{preview.label}</strong>
                    </div>
                    <div className="evaluation-image-lightbox-actions">
                        <button
                            type="button"
                            className="image-lightbox-close evaluation-image-zoom-button"
                            onClick={() => changeZoom(-1)}
                            disabled={safeZoom <= 1}
                            aria-label="Alejar imagen"
                            title="Alejar"
                        >
                            <ZoomOutIcon />
                        </button>
                        <output aria-live="polite">{Math.round(safeZoom * 100)}%</output>
                        <button
                            type="button"
                            className="image-lightbox-close evaluation-image-zoom-button"
                            onClick={() => changeZoom(1)}
                            disabled={safeZoom >= 4}
                            aria-label="Acercar imagen"
                            title="Acercar"
                        >
                            <ZoomInIcon />
                        </button>
                        <button
                            type="button"
                            className="evaluation-image-reset-zoom"
                            onClick={resetView}
                            disabled={safeZoom === 1 && pan.x === 0 && pan.y === 0}
                        >
                            Restablecer
                        </button>
                        <button
                            type="button"
                            className="image-lightbox-close"
                            onClick={onClose}
                            aria-label="Cerrar vista ampliada"
                        >
                            <CloseIcon />
                        </button>
                    </div>
                </header>
                <div
                    ref={viewportRef}
                    className={`image-lightbox-content evaluation-image-lightbox-content detection-image-lightbox-content${
                        safeZoom > 1 ? ' is-zoomed' : ''
                    }${isDragging ? ' is-dragging' : ''}`}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={finishPointerDrag}
                    onPointerCancel={finishPointerDrag}
                >
                    <div
                        ref={canvasRef}
                        className={`detection-image-lightbox-canvas tone-${preview.variant ?? 'neutral'}`}
                        style={{
                            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${safeZoom})`,
                        }}
                    >
                        <img
                            src={preview.url}
                            alt={preview.alt}
                            draggable="false"
                            onLoad={(event) => setDimensions({
                                width: event.currentTarget.naturalWidth,
                                height: event.currentTarget.naturalHeight,
                            })}
                        />
                        <DetectionBoxesOverlay
                            detections={preview.detections}
                            dimensions={dimensions}
                        />
                    </div>
                </div>
                <footer>
                    <span>
                        Usa + / − para cambiar el zoom. Con zoom activo, arrastra la imagen para desplazarte.
                    </span>
                </footer>
            </dialog>
        </div>
    )

    return createPortal(modal, document.body)
}

function DetectionImageCard({
    evaluation,
    result,
    eigenCamEntry,
    onGenerateEigenCam,
}) {
    const [previewImage, setPreviewImage] = useState(null)
    const [showEigenCam, setShowEigenCam] = useState(true)
    const eigenCamLoading = eigenCamEntry.status === 'loading'
    const eigenCamReady = eigenCamEntry.status === 'success'
    const eigenCamError = eigenCamEntry.status === 'error'
        ? eigenCamEntry.error
        : ''
    const originalImagePath = (
        `/datasets/${evaluation.dataset_id}/images/${result.image_id}`
    )

    function handleEigenCamAction() {
        if (eigenCamReady) {
            setShowEigenCam((current) => !current)
            return
        }
        onGenerateEigenCam(evaluation.result_id, result.image_id)
    }

    return (
        <article className="detection-image-result-card">
            <header>
                <div>
                    <strong>{result.filename}</strong>
                    <small>{result.image_id}</small>
                </div>
                <span className={`detection-legacy-badge ${result.attack_success ? 'success' : 'neutral'}`}>
                    {result.attack_success ? 'Indicador heurístico positivo' : 'Indicador heurístico negativo'}
                </span>
            </header>

            <div className="detection-preview-grid">
                <DetectionOverlayImage
                    imagePath={originalImagePath}
                    result={result}
                    variant="original"
                    onExpand={setPreviewImage}
                />
                <DetectionOverlayImage
                    imagePath={result.adversarial_image_url}
                    result={result}
                    variant="adversarial"
                    onExpand={setPreviewImage}
                />
            </div>

            <p className="detection-preview-note">
                Las dos vistas conservan la geometría fuente. La segunda imagen es el resultado real del ataque, no una proyección
                sobre la original.
            </p>

            <dl className="detection-image-result-metrics">
                <div><dt>Detecciones originales</dt><dd>{formatInteger(result.original_detection_count)}</dd></div>
                <div><dt>Detecciones tras el ataque</dt><dd>{formatInteger(result.adversarial_detection_count)}</dd></div>
                <div><dt>Δ detecciones</dt><dd>{formatInteger(result.detections_difference)}</dd></div>
                <div><dt>Conf. original</dt><dd>{formatDecimal(result.average_original_confidence)}</dd></div>
                <div><dt>Conf. tras el ataque</dt><dd>{formatDecimal(result.average_adversarial_confidence)}</dd></div>
                <div><dt>Δ confianza</dt><dd>{formatDecimal(result.average_confidence_difference)}</dd></div>
            </dl>

            <section className="detection-eigencam-section">
                <div className="detection-eigencam-action">
                    <div>
                        <strong>Explicabilidad Eigen-CAM</strong>
                        <span>
                            Genera los mapas original y tras el ataque solo para esta imagen.
                        </span>
                    </div>
                    <button
                        type="button"
                        className="evaluation-gradcam-button"
                        disabled={eigenCamLoading}
                        aria-expanded={eigenCamReady ? showEigenCam : undefined}
                        onClick={handleEigenCamAction}
                    >
                        {getDetectionEigenCamButtonLabel(
                            eigenCamEntry,
                            showEigenCam,
                        )}
                    </button>
                </div>

                {eigenCamLoading && (
                    <output className="evaluation-gradcam-status">
                        Calculando Eigen-CAM sin repetir el ataque ni las detecciones.
                    </output>
                )}

                {eigenCamError && (
                    <div className="evaluation-gradcam-error" role="alert">
                        {eigenCamError}
                    </div>
                )}

                {eigenCamReady && showEigenCam && (
                    <div className="detection-eigencam-grid">
                        <EigenCamPreview
                            label="Eigen-CAM original"
                            path={eigenCamEntry.data.original_eigencam_image_url}
                            filename={result.filename}
                            variant="original"
                            onExpand={setPreviewImage}
                        />
                        <EigenCamPreview
                            label="Eigen-CAM tras el ataque"
                            path={eigenCamEntry.data.adversarial_eigencam_image_url}
                            filename={result.filename}
                            variant="adversarial"
                            onExpand={setPreviewImage}
                        />
                    </div>
                )}
            </section>

            <details className="detection-prediction-list">
                <summary>Ver predicciones en texto</summary>
                <dl>
                    <div><dt>Originales</dt><dd>{formatDetections(result.original_detections)}</dd></div>
                    <div><dt>Tras el ataque</dt><dd>{formatDetections(result.adversarial_detections)}</dd></div>
                </dl>
            </details>

            {previewImage && (
                <DetectionImageLightbox
                    key={`${previewImage.url}:${previewImage.variant ?? 'neutral'}`}
                    preview={previewImage}
                    onClose={() => setPreviewImage(null)}
                />
            )}
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
            <section className="evaluation-surface detection-table-surface detection-images-section">
                <div className="evaluation-section-heading detection-images-heading">
                    <div>
                        <span className="evaluation-eyebrow">Exploración visual</span>
                        <h3>Predicciones por imagen</h3>
                        <p>
                            Compara los píxeles y cajas originales y tras el ataque, y
                            solicita Eigen-CAM únicamente para los casos relevantes. Las
                            cajas usan el filtro público histórico de confianza &gt; 0.25.
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
