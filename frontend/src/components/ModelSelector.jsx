import { useEffect, useState } from 'react'
import { getModels, selectModel } from '../api/models'

function ModelSelector() {
    const [models, setModels] = useState([])
    const [selectedModelId, setSelectedModelId] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function loadModels() {
        try {
            setError('')
            const data = await getModels()
            setModels(data)

            const activeModel = data.find((model) => model.active)
            if (activeModel) {
                setSelectedModelId(activeModel.id)
            }
        } catch (err) {
            setError(err.message)
        }
    }

    useEffect(() => {
        loadModels()
    }, [])

    async function handleModelChange(event) {
        const modelId = event.target.value

        try {
            setLoading(true)
            setError('')

            const selectedModel = await selectModel(modelId)

            setSelectedModelId(selectedModel.id)

            setModels((currentModels) =>
                currentModels.map((model) => ({
                    ...model,
                    active: model.id === selectedModel.id,
                    loaded: model.id === selectedModel.id,
                })),
            )
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const activeModel = models.find((model) => model.active)

    return (
        <div>
            <h2>Modelo</h2>

            {error && <p className="error">{error}</p>}

            {models.length === 0 ? (
                <p>No hay modelos disponibles.</p>
            ) : (
                <select
                    value={selectedModelId}
                    onChange={handleModelChange}
                    disabled={loading}
                >
                    {models.map((model) => (
                        <option key={model.id} value={model.id}>
                            {model.name}
                            {model.active ? ' (activo)' : ''}
                            {model.loaded ? ' (cargado)' : ' (no cargado)'}
                        </option>
                    ))}
                </select>
            )}

            {activeModel && (
                <div>
                    <p>
                        Modelo activo: <strong>{activeModel.name}</strong>
                    </p>
                    <p>
                        Estado:{' '}
                        <strong>{activeModel.loaded ? 'Cargado en memoria' : 'No cargado'}</strong>
                    </p>
                </div>
            )}
        </div>
    )
}

export default ModelSelector