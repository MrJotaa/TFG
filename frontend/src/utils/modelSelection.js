export function activateModelInList(models, selectedModelId) {
    return models.map((model) => ({
        ...model,
        active: model.id === selectedModelId,
        loaded: model.id === selectedModelId,
    }))
}
