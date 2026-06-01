import { useEffect, useState } from 'react'
import { getDatasetImages, updateImageLabel } from '../api/datasets'
import ImageLabelEditor from './ImageLabelEditor'

function ImageGallery({ selectedDataset, selectedImage, onSelectImage, refreshKey }) {
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
        <div>
            <h2>Imágenes</h2>

            {error && <p className="error">{error}</p>}

            {!selectedDataset ? (
                <p>Selecciona un dataset para ver sus imágenes.</p>
            ) : images.length === 0 ? (
                <p>Este dataset no tiene imágenes.</p>
            ) : (
                <div>
                    {images.map((image) => (
                        <div
                            key={image.id}
                            style={{
                                display: 'inline-block',
                                margin: '8px',
                                verticalAlign: 'top',
                            }}
                        >
                            <img
                                src={`http://127.0.0.1:8000${image.url}`}
                                alt={image.filename}
                                width={120}
                                style={{
                                    margin: '5px',
                                    border:
                                        selectedImage?.id === image.id
                                            ? '3px solid blue'
                                            : '1px solid #ccc',
                                    cursor: 'pointer',
                                }}
                                onClick={() => onSelectImage(image)}
                            />

                            <ImageLabelEditor
                                image={image}
                                onSaveLabel={handleLabelChange}
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

export default ImageGallery