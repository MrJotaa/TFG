import { useState } from 'react'
import { addImagesToDataset } from '../api/datasets'

function DatasetImageUploader({ selectedDataset, onImagesUploaded }) {
    const [files, setFiles] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function handleSubmit(event) {
        event.preventDefault()

        if (!selectedDataset) {
            setError('Debes seleccionar un dataset antes de añadir imágenes.')
            return
        }

        if (files.length === 0) {
            setError('Debes seleccionar al menos una imagen.')
            return
        }

        try {
            setLoading(true)
            setError('')

            const uploadedImages = await addImagesToDataset({
                datasetId: selectedDataset.id,
                files,
            })

            onImagesUploaded(uploadedImages)

            setFiles([])
            event.target.reset()
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <h2>Añadir imágenes al dataset</h2>

            {!selectedDataset && (
                <p>Selecciona un dataset para poder añadir imágenes.</p>
            )}

            {selectedDataset && (
                <p>
                    Dataset seleccionado: <strong>{selectedDataset.name}</strong>
                </p>
            )}

            {error && <p className="error">{error}</p>}

            <form onSubmit={handleSubmit}>
                <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(event) => setFiles(event.target.files)}
                    disabled={!selectedDataset}
                />

                <button type="submit" disabled={!selectedDataset || loading}>
                    {loading ? 'Añadiendo...' : 'Añadir imágenes'}
                </button>
            </form>
        </div>
    )
}

export default DatasetImageUploader