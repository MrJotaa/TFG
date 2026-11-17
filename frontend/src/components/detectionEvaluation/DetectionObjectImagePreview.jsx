import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
    DetectionImageLightbox,
    DetectionOverlayImage,
} from './DetectionImageResults'
import { formatDecimal, formatInteger } from '../../utils/metricFormatting'

const COUNTERS = [
    ['total_objects', 'Objetos evaluados'],
    ['preserved_count', 'Preservados'],
    ['suppressed_count', 'Suprimidos'],
    ['recovered_count', 'Recuperados'],
    ['missed_both_count', 'No detectados en ambas'],
]

const OUTCOME_DETAILS = {
    preserved: {
        title: 'Preservado',
        description: 'La referencia COCO tiene una detección emparejada válida tanto en la imagen original como después del ataque.',
    },
    suppressed: {
        title: 'Suprimido',
        description: 'La referencia COCO tenía una detección emparejada originalmente, pero no después del ataque.',
    },
    recovered: {
        title: 'Recuperado',
        description: 'La referencia COCO no tenía una detección emparejada originalmente, pero sí después del ataque.',
    },
    missed_both: {
        title: 'No detectado en ambas',
        description: 'La referencia COCO no tiene una detección emparejada válida en ninguno de los dos resultados.',
    },
}

function DetectionObjectImagePreview({ image, onShowImage, onClose }) {
    const [enlarged, setEnlarged] = useState(null)
    const closeButtonRef = useRef(null)

    useEffect(() => {
        const previousOverflow = document.body.style.overflow
        const previousFocus = document.activeElement
        document.body.style.overflow = 'hidden'
        closeButtonRef.current?.focus()
        return () => {
            document.body.style.overflow = previousOverflow
            previousFocus?.focus?.()
        }
    }, [])

    useEffect(() => {
        function handleEscape(event) {
            if (event.key === 'Escape' && !enlarged) onClose()
        }
        window.addEventListener('keydown', handleEscape)
        return () => window.removeEventListener('keydown', handleEscape)
    }, [enlarged, onClose])

    const overlayResult = {
        filename: image.filename ?? image.imageId,
        original_detections: image.originalDetections,
        adversarial_detections: image.adversarialDetections,
    }
    const title = image.filename ?? `Imagen ${image.imageId}`
    const selected = image.selectedObject
    const selectedOutcome = selected ? OUTCOME_DETAILS[selected.outcome] : null
    const category = selected?.category?.category_name ?? 'Objeto'
    const hasOriginal = Boolean(selected?.original_observation)
    const hasAdversarial = Boolean(selected?.adversarial_observation)

    return createPortal(
        <>
            <div className="image-lightbox-backdrop detection-object-image-backdrop">
                <button
                    type="button"
                    className="modal-backdrop-dismiss"
                    aria-label="Cerrar inspección de imagen"
                    tabIndex={-1}
                    onClick={onClose}
                />
                <dialog
                    open
                    className="image-lightbox detection-object-image-dialog"
                    aria-modal="true"
                    aria-label={`Inspección por objeto: ${title}`}
                >
                    <header>
                        <div>
                            <span className="evaluation-eyebrow">{selected ? 'Explorador de objetos · Objeto seleccionado' : 'Resultados por imagen · Análisis por objeto'}</span>
                            <strong>{selected ? `${category} · ${title}` : title}</strong>
                            <small>{selected ? `ID de objeto: ${selected.object_id} · ` : ''}ID de imagen: {image.imageId}</small>
                        </div>
                        <div className="detection-object-image-header-actions">
                            {selected && (
                                <button
                                    type="button"
                                    className="detection-object-image-close"
                                    onClick={onShowImage}
                                >
                                    Ver imagen completa
                                </button>
                            )}
                            <button
                                ref={closeButtonRef}
                                type="button"
                                className="detection-object-image-close"
                                onClick={onClose}
                                aria-label="Cerrar inspección de imagen"
                            >
                                Cerrar
                            </button>
                        </div>
                    </header>
                    <div className="detection-object-image-content">
                        {selected ? (
                            <section className="detection-object-inspection-summary" aria-label="Resultado del objeto seleccionado">
                                <div className="detection-object-inspection-heading">
                                    <strong>{selectedOutcome?.title ?? selected.outcome}</strong>
                                    <span>{category}</span>
                                </div>
                                <p>{selectedOutcome?.description ?? 'Resultado del emparejamiento de esta referencia COCO.'}</p>
                                <dl className="detection-object-inspection-observations">
                                    <div>
                                        <dt>Original</dt>
                                        <dd>{hasOriginal
                                            ? `Confianza ${formatDecimal(selected.original_observation.confidence)} · IoU ${formatDecimal(selected.original_observation.iou_with_ground_truth)}`
                                            : 'Sin detección emparejada'}</dd>
                                    </div>
                                    <div>
                                        <dt>Tras el ataque</dt>
                                        <dd>{hasAdversarial
                                            ? `Confianza ${formatDecimal(selected.adversarial_observation.confidence)} · IoU ${formatDecimal(selected.adversarial_observation.iou_with_ground_truth)}`
                                            : 'Sin detección emparejada'}</dd>
                                    </div>
                                </dl>
                                <p className="detection-object-inspection-note">
                                    Sin emparejamiento no significa necesariamente sin predicciones:
                                    pueden existir cajas que no cumplan los umbrales de confianza,
                                    IoU o categoría. La pestaña «Imágenes» permite inspeccionarlas.
                                </p>
                            </section>
                        ) : (
                            <p className="detection-object-image-explanation">
                                Las cajas discontinuas representan las referencias COCO evaluables.
                                Las cajas continuas indican detecciones emparejadas con esos objetos
                                bajo los umbrales de la evaluación. No se muestran aquí todas las
                                predicciones de la imagen; estas se consultan en «Imágenes».
                            </p>
                        )}
                        {!selected && image.aggregate && (
                            <dl className="detection-object-image-counts">
                                {COUNTERS.map(([field, label]) => (
                                    <div key={field}>
                                        <dt>{label}</dt>
                                        <dd>{formatInteger(image.aggregate[field])}</dd>
                                    </div>
                                ))}
                            </dl>
                        )}
                        <div className="detection-object-image-legend" aria-label="Leyenda de cajas">
                            <span className="reference">Referencia COCO</span>
                            <span className="original">Detección original emparejada</span>
                            <span className="adversarial">Detección adversaria emparejada</span>
                        </div>
                        <div className="detection-preview-grid detection-object-image-grid">
                            <DetectionOverlayImage
                                key={`original:${image.imageId}`}
                                imagePath={image.originalImagePath}
                                result={overlayResult}
                                variant="original"
                                onExpand={setEnlarged}
                            />
                            <DetectionOverlayImage
                                key={`adversarial:${image.imageId}`}
                                imagePath={image.adversarialImagePath}
                                result={overlayResult}
                                variant="adversarial"
                                onExpand={setEnlarged}
                            />
                        </div>
                        {!image.objects.length && (
                            <p className="detection-object-image-explanation">
                                El resultado conservado no contiene cajas de referencia individuales
                                para esta imagen.
                            </p>
                        )}
                    </div>
                </dialog>
            </div>
            {enlarged && (
                <DetectionImageLightbox
                    key={`${enlarged.variant}:${enlarged.url}`}
                    preview={enlarged}
                    onClose={() => setEnlarged(null)}
                />
            )}
        </>,
        document.body,
    )
}

export default DetectionObjectImagePreview
