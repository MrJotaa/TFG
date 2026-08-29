import { useEffect, useState } from 'react'
import { getDatasetImages } from '../api/datasets'
import { filterVisibleImages } from '../utils/imageGallery'

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


function GalleryEmptyState({ workspace, title, copy, fallback }) {
    if (workspace) {
        return (
            <div className="dashboard-empty-state">
                <strong>{title}</strong>
                <span>{copy}</span>
            </div>
        )
    }
    return <p>{fallback}</p>
}

function GalleryImageCard({ image, selectedImage, onSelectImage, workspace, compact, allowPreview, onPreview }) {
    const isSelected = selectedImage?.id === image.id
    let caption = null
    if (workspace) {
        caption = <p className="workspace-image-name" title={image.filename}>{image.filename}</p>
    } else if (!compact) {
        caption = <p className="image-filename">{image.filename}</p>
    }

    return (
        <div className={`image-card${isSelected ? ' selected' : ''}`}>
            <div className="image-card-visual">
                <button type="button" className="image-button" onClick={() => onSelectImage(image)} aria-pressed={isSelected} title={`Seleccionar ${image.filename}`}>
                    <img src={imageSource(image)} alt={image.filename} />
                </button>
                {workspace && isSelected && <span className="image-selection-mark" aria-label="Imagen seleccionada"><CheckIcon /></span>}
                {allowPreview && (
                    <button type="button" className="image-preview-button" onClick={() => onPreview(image)} title={`Ampliar ${image.filename}`} aria-label={`Ampliar ${image.filename}`}>
                        <ExpandIcon />
                    </button>
                )}
            </div>
            {caption}
        </div>
    )
}

function GalleryContent(props) {
    const { selectedDataset, images, visibleImages, workspace } = props
    if (!selectedDataset) {
        return <GalleryEmptyState workspace={workspace} title="Selecciona un conjunto de datos" copy="Sus imágenes aparecerán aquí para comenzar el análisis." fallback="Selecciona un conjunto de datos para ver sus imágenes." />
    }
    if (images.length === 0) {
        return <GalleryEmptyState workspace={workspace} title="Conjunto vacío" copy="Este conjunto de datos todavía no contiene imágenes." fallback="Este conjunto de datos no tiene imágenes." />
    }
    if (visibleImages.length === 0) {
        return <div className="dashboard-empty-state"><strong>No hay imágenes en este filtro</strong><span>Prueba otra vista para seguir gestionando el conjunto de datos.</span></div>
    }
    return (
        <div className="image-grid">
            {visibleImages.map((image) => <GalleryImageCard key={image.id} image={image} {...props} onPreview={props.setPreviewImage} />)}
        </div>
    )
}

function ImageGallery({
    selectedDataset,
    selectedImage,
    onSelectImage,
    refreshKey,
    compact = false,
    workspace = false,
    allowPreview = false,
    onImagesLoaded,
    filterImage,
}) {
    const [images, setImages] = useState([])
    const [error, setError] = useState('')
    const [previewImage, setPreviewImage] = useState(null)

    useEffect(() => {
        let active = true

        if (!selectedDataset) {
            Promise.resolve().then(() => {
                if (!active) return
                setImages([])
                setPreviewImage(null)
                onImagesLoaded?.([])
            })
            return () => { active = false }
        }

        getDatasetImages(selectedDataset.id)
            .then((data) => {
                if (!active) return
                setError('')
                setImages(data)
                onImagesLoaded?.(data)
            })
            .catch((requestError) => {
                if (active) setError(requestError.message)
            })

        return () => { active = false }
    }, [selectedDataset, refreshKey, onImagesLoaded])

    useEffect(() => {
        if (!previewImage) return undefined

        function handleKeyDown(event) {
            if (event.key === 'Escape') setPreviewImage(null)
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [previewImage])

    const visibleImages = filterVisibleImages(images, filterImage)
    const galleryClassName = [
        'image-gallery',
        compact ? 'compact' : '',
        workspace ? 'workspace-image-gallery' : '',
    ].filter(Boolean).join(' ')

    return (
        <div className={galleryClassName}>
            {!compact && !workspace && <h2>Imágenes</h2>}
            {error && <p className="error">{error}</p>}

            <GalleryContent
                selectedDataset={selectedDataset}
                images={images}
                visibleImages={visibleImages}
                workspace={workspace}
                compact={compact}
                allowPreview={allowPreview}
                selectedImage={selectedImage}
                onSelectImage={onSelectImage}
                setPreviewImage={setPreviewImage}
            />

            {previewImage && (
                <div className="image-lightbox-backdrop">
                    <button
                        type="button"
                        className="modal-backdrop-dismiss"
                        tabIndex={-1}
                        aria-label="Cerrar vista ampliada"
                        onClick={() => setPreviewImage(null)}
                    />
                    <dialog
                        open
                        className="image-lightbox"
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
                    </dialog>
                </div>
            )}
        </div>
    )
}

export default ImageGallery
