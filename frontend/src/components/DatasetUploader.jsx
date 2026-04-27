import { useState } from 'react'
import { uploadDataset } from '../api/datasets'

function DatasetUploader({ selectedUser, onDatasetUploaded }) {
    const [name, setName] = useState('')
    const [files, setFiles] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function handleSubmit(event) {
        event.preventDefault()

        if (!selectedUser) {
            setError('Debes seleccionar un usuario antes de subir un dataset.')
            return
        }

        if (files.length === 0) {
            setError('Debes seleccionar al menos una imagen.')
            return
        }

        try {
            setLoading(true)
            setError('')

            const dataset = await uploadDataset({
                userId: selectedUser.id,
                name,
                files,
            })

            onDatasetUploaded(dataset)
            setName('')
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
            <h2>Subir dataset</h2>

            {!selectedUser && (
                <p>Selecciona un usuario antes de subir un dataset.</p>
            )}

            {error && <p className="error">{error}</p>}

            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Nombre del dataset"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    required
                    disabled={!selectedUser}
                />

                <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(event) => setFiles(event.target.files)}
                    disabled={!selectedUser}
                />

                <button type="submit" disabled={!selectedUser || loading}>
                    {loading ? 'Subiendo...' : 'Subir dataset'}
                </button>
            </form>
        </div>
    )
}

export default DatasetUploader