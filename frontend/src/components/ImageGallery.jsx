import { useEffect, useState } from 'react'
import { getDatasetImages } from '../api/datasets'

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
                        <img
                            key={image.id}
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
                    ))}
                </div>
            )}
        </div>
    )
}

export default ImageGallery