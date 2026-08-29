import { useEffect, useState } from 'react'
import { getModels, selectModel } from '../api/models'
import { activateModelInList } from '../utils/modelSelection'

function ModelSelector({ onActiveModelChange }) {
    const [models, setModels] = useState([])
    const [selectedModelId, setSelectedModelId] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    useEffect(() => {
        let cancelled = false

        getModels()
            .then((data) => {
                if (cancelled) return

                setError('')
                setModels(data)

                const activeModel = data.find((model) => model.active)
                if (activeModel) {
                    setSelectedModelId(activeModel.id)
                    onActiveModelChange?.(activeModel)
                }
            })
            .catch((err) => {
                if (!cancelled) setError(err.message)
            })

        return () => {
            cancelled = true
        }
    }, [onActiveModelChange])

    const selectedModel = models.find((model) => model.id === selectedModelId)
    const selectedModelIsActive = Boolean(selectedModel?.active)

    async function handleSelectModel() {
        if (!selectedModelId || selectedModelIsActive) return

        try {
            setLoading(true)
            setError('')

            const activatedModel = await selectModel(selectedModelId)

            setModels((currentModels) =>
                activateModelInList(currentModels, activatedModel.id),
            )

            onActiveModelChange?.({
                ...activatedModel,
                loaded: true,
            })
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    let activateButtonLabel = 'Activar modelo'
    if (loading) activateButtonLabel = 'Activando…'
    else if (selectedModelIsActive) activateButtonLabel = 'Modelo activo'

    return (
        <section className="sidebar-block dashboard-config-section">
            <div className="dashboard-config-heading">
                <div>
                    <span className="dashboard-section-eyebrow">Configuración</span>
                    <h2>Modelo</h2>
                </div>
                <span className="dashboard-config-step">01</span>
            </div>

            {error && <p className="error">{error}</p>}

            {models.length === 0 ? (
                <p className="dashboard-empty-copy">No hay modelos disponibles.</p>
            ) : (
                <div className="sidebar-list model-list" aria-label="Modelos disponibles">
                    {models.map((model) => {
                        const isSelected = selectedModelId === model.id
                        return (
                            <button
                                key={model.id}
                                type="button"
                                className={isSelected ? 'sidebar-list-item active' : 'sidebar-list-item'}
                                onClick={() => setSelectedModelId(model.id)}
                                disabled={loading}
                                aria-pressed={isSelected}
                            >
                                <span className="model-option-copy">
                                    <strong>{model.name}</strong>
                                    {model.task_type && (
                                        <small className="model-task-badge">
                                            {model.task_type === 'classification' ? 'Clasificación' : 'Detección'}
                                        </small>
                                    )}
                                </span>

                                {model.active && (
                                    <span className="model-active-indicator">
                                        <span aria-hidden="true" />{' '}
                                        Activo
                                    </span>
                                )}
                            </button>
                        )
                    })}
                </div>
            )}

            <button
                type="button"
                className="secondary-action dashboard-activate-model"
                onClick={handleSelectModel}
                disabled={!selectedModelId || selectedModelIsActive || loading}
            >
                {activateButtonLabel}
            </button>
        </section>
    )
}

export default ModelSelector
