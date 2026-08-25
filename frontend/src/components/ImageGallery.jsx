import { useEffect, useState } from 'react'
import {
    deleteDatasetImage,
    getDatasetImages,
    updateImageLabel,
} from '../api/datasets'
import ImageLabelEditor from './ImageLabelEditor'
import { getAnnotationStatusLabel } from '../utils/cocoAnnotations'

function CheckIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m6.5 12.3 3.4 3.4 7.6-7.6" />
        </svg>
    )
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

function imageSource(image) {
    return `http://127.0.0.1:8000${image.url}`
}

function ImageGallery({
    selectedDataset,
    selectedImage,
    onSelectImage,
    refreshKey,
    compact = false,
    workspace = false,
    allowPreview = false,
    showLabelEditor = true,
    showDeleteButton = false,
    onImageDeleted,
}) {
    const [images, setImages] = useState([])
    const [error, setError] = useState('')
    const [previewImage, setPreviewImage] = useState(null)

    useEffect(() => {
        async function loadImages() {
            if (!selectedDataset) {
                setImages([])
                setPreviewImage(null)
                return
            }

            try {
                setError('')
                const data = await getDatasetImages(selectedDataset.id)
                setImages(data)
            } catch (err) {
                setError(err.message)
            }
        }

        loadImages()
    }, [selectedDataset, refreshKey])

    useEffect(() => {
        if (!previewImage) return undefined

        function handleKeyDown(event) {
            if (event.key === 'Escape') setPreviewImage(null)
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [previewImage])

    async function handleLabelChange(imageId, label) {
        if (!selectedDataset) return

        try {
            setError('')

            const updatedImage = await updateImageLabel({
                datasetId: selectedDataset.id,
                imageId,
                label,
            })

            setImages((currentImages) =>
                currentImages.map((image) =>
                    image.id === imageId
                        ? { ...image, label: updatedImage.label }
                        : image,
                ),
            )
        } catch (err) {
            setError(err.message)
        }
    }

    async function handleDeleteImage(imageId) {
        if (!selectedDataset) return

        const confirmed = window.confirm(
            '¿Seguro que quieres borrar esta imagen? Esta acción no se puede deshacer.',
        )

        if (!confirmed) return

        try {
            setError('')

            await deleteDatasetImage({
                datasetId: selectedDataset.id,
                imageId,
            })

            setImages((currentImages) =>
                currentImages.filter((image) => image.id !== imageId),
            )

            if (selectedImage?.id === imageId) {
                onSelectImage(null)
            }

            onImageDeleted?.(imageId)
        } catch (err) {
            setError(err.message)
        }
    }

    const galleryClassName = [
        'image-gallery',
        compact ? 'compact' : '',
        workspace ? 'workspace-image-gallery' : '',
    ].filter(Boolean).join(' ')

    return (
        <div className={galleryClassName}>
            {!compact && !workspace && <h2>Imágenes</h2>}

            {error && <p className="error">{error}</p>}

            {!selectedDataset ? (
                workspace ? (
                    <div className="dashboard-empty-state">
                        <strong>Selecciona un conjunto de datos</strong>
                        <span>Sus imágenes aparecerán aquí para comenzar el análisis.</span>
                    </div>
                ) : (
                    <p>Selecciona un conjunto de datos para ver sus imágenes.</p>
                )
            ) : images.length === 0 ? (
                workspace ? (
                    <div className="dashboard-empty-state">
                        <strong>Dataset vacío</strong>
                        <span>Este conjunto de datos todavía no contiene imágenes.</span>
                    </div>
                ) : (
                    <p>Este conjunto de datos no tiene imágenes.</p>
                )
            ) : (
                <div className="image-grid">
                    {images.map((image) => {
                        const isSelected = selectedImage?.id === image.id
                        return (
                            <div
                                key={image.id}
                                className={`image-card${isSelected ? ' selected' : ''}`}
                            >
                                <div className="image-card-visual">
                                    <button
                                        type="button"
                                        className="image-button"
                                        onClick={() => onSelectImage(image)}
                                        aria-pressed={isSelected}
                                        title={`Seleccionar ${image.filename}`}
                                    >
                                        <img
                                            src={imageSource(image)}
                                            alt={image.filename}
                                        />
                                    </button>

                                    {workspace && isSelected && (
                                        <span className="image-selection-mark" aria-label="Imagen seleccionada">
                                            <CheckIcon />
                                        </span>
                                    )}

                                    {allowPreview && (
                                        <button
                                            type="button"
                                            className="image-preview-button"
                                            onClick={() => setPreviewImage(image)}
                                            title={`Ampliar ${image.filename}`}
                                            aria-label={`Ampliar ${image.filename}`}
                                        >
                                            <ExpandIcon />
                                        </button>
                                    )}
                                </div>

                                {workspace ? (
                                    <p className="workspace-image-name" title={image.filename}>
                                        {image.filename}
                                    </p>
                                ) : !compact ? (
                                    <>
                                        <p className="image-filename">{image.filename}</p>
                                        <span
                                            className={`annotation-status ${image.annotation_review_status || 'unreviewed'}`}
                                        >
                                            {getAnnotationStatusLabel(
                                                image.annotation_review_status,
                                            )}
                                        </span>
                                    </>
                                ) : null}

                                {showLabelEditor && (
                                    <ImageLabelEditor
                                        image={image}
                                        onSaveLabel={handleLabelChange}
                                    />
                                )}

                                {showDeleteButton && (
                                    <button
                                        type="button"
                                        className="danger-action small full-width"
                                        onClick={() => handleDeleteImage(image.id)}
                                    >
                                        Borrar
                                    </button>
                                )}
                            </div>
                        )
                    })}
                </div>
            )}

            {previewImage && (
                <div
                    className="image-lightbox-backdrop"
                    role="presentation"
                    onMouseDown={(event) => {
                        if (event.target === event.currentTarget) setPreviewImage(null)
                    }}
                >
                    <section
                        className="image-lightbox"
                        role="dialog"
                        aria-modal="true"
                        aria-label={`Vista ampliada de ${previewImage.filename}`}
                    >
                        <header>
                            <div>
                                <span className="dashboard-section-eyebrow">Vista ampliada</span>
                                <strong>{previewImage.filename}</strong>
                            </div>
                            <button
                                type="button"
                                className="image-lightbox-close"
                                onClick={() => setPreviewImage(null)}
                                aria-label="Cerrar vista ampliada"
                            >
                                <CloseIcon />
                            </button>
                        </header>
                        <div className="image-lightbox-content">
                            <img src={imageSource(previewImage)} alt={previewImage.filename} />
                        </div>
                        {previewImage.label && (
                            <footer>
                                <span>Etiqueta</span>
                                <strong>{previewImage.label}</strong>
                            </footer>
                        )}
                    </section>
                </div>
            )}
        </div>
    )
}

export default ImageGallery
