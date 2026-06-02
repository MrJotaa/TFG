import { useEffect, useState } from 'react'
import { getDatasetImages, updateImageLabel } from '../api/datasets'
import ImageLabelEditor from './ImageLabelEditor'

function ImageGallery({
    selectedDataset,
    selectedImage,
    onSelectImage,
    refreshKey,
    compact = false,
    showLabelEditor = true,
}) {
    const [images, setImages] = useState([])
    const [error, setError] = useState('')

    useEffect(() => {
        async function loadImages() {
            if (!selectedDataset) {
                setImages([])
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

    return (
        <div className={compact ? 'image-gallery compact' : 'image-gallery'}>
            {!compact && <h2>Imágenes</h2>}

            {error && <p className="error">{error}</p>}

            {!selectedDataset ? (
                <p>Selecciona un dataset para ver sus imágenes.</p>
            ) : images.length === 0 ? (
                <p>Este dataset no tiene imágenes.</p>
            ) : (
                <div className="image-grid">
                    {images.map((image) => (
                        <div
                            key={image.id}
                            className={
                                selectedImage?.id === image.id
                                    ? 'image-card selected'
                                    : 'image-card'
                            }
                        >
                            <button
                                type="button"
                                className="image-button"
                                onClick={() => onSelectImage(image)}
                            >
                                <img
                                    src={`http://127.0.0.1:8000${image.url}`}
                                    alt={image.filename}
                                />
                            </button>

                            {!compact && (
                                <p className="image-filename">{image.filename}</p>
                            )}

                            {showLabelEditor && (
                                <ImageLabelEditor
                                    image={image}
                                    onSaveLabel={handleLabelChange}
                                />
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

export default ImageGallery