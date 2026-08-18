import { useEffect, useState } from 'react'
import { getModels, selectModel } from '../api/models'
import { activateModelInList } from '../utils/modelSelection'

function ModelSelector({ onActiveModelChange }) {
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
                onActiveModelChange?.(activeModel)
            }
        } catch (err) {
            setError(err.message)
        }
    }

    useEffect(() => {
        loadModels()
    }, [])

    async function handleSelectModel() {
        if (!selectedModelId) return

        try {
            setLoading(true)
            setError('')

            const selectedModel = await selectModel(selectedModelId)

            setModels((currentModels) =>
                activateModelInList(currentModels, selectedModel.id),
            )

            onActiveModelChange?.({
                ...selectedModel,
                loaded: true,
            })
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="sidebar-block">
            <div className="sidebar-block-title">Modelos</div>

            {error && <p className="error">{error}</p>}

            {models.length === 0 ? (
                <p>No hay modelos disponibles.</p>
            ) : (
                <div className="sidebar-list">
                    {models.map((model) => (
                        <button
                            key={model.id}
                            type="button"
                            className={
                                selectedModelId === model.id
                                    ? 'sidebar-list-item active'
                                    : 'sidebar-list-item'
                            }
                            onClick={() => setSelectedModelId(model.id)}
                            disabled={loading}
                        >
                            <div>
                                <span>{model.name}</span>
                                {model.task_type && (
                                    <small className="model-task-badge">
                                        {model.task_type === 'classification' ? 'Clasificación' : 'Detección'}
                                    </small>
                                )}
                            </div>

                            {model.active && <small>Activo</small>}
                        </button>
                    ))}
                </div>
            )}

            <button
                type="button"
                className="secondary-action"
                onClick={handleSelectModel}
                disabled={!selectedModelId || loading}
            >
                {loading ? 'Seleccionando...' : 'Select'}
            </button>
        </div>
    )
}

export default ModelSelector