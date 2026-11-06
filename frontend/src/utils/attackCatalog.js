export function getAttackDefinition(attackCatalog, attackId) {
    return (attackCatalog ?? []).find((attack) => attack.id === attackId) ?? null
}

function supportsAttackModel(attack, modelId) {
    return !modelId || !attack.supported_model_ids
        || attack.supported_model_ids.includes(modelId)
}

export function isAttackSupportedForTask(attackCatalog, attackId, taskType, modelId = null) {
    const attack = getAttackDefinition(attackCatalog, attackId)
    return Boolean(
        attack?.supported_task_types?.includes(taskType)
        && supportsAttackModel(attack, modelId),
    )
}

export function getSupportedAttacksForTask(attackCatalog, taskType, modelId = null) {
    return (attackCatalog ?? []).filter((attack) =>
        attack.supported_task_types?.includes(taskType)
        && supportsAttackModel(attack, modelId),
    )
}

export function resolveCompatibleAttackSelection(
    attackCatalog,
    activeModel,
    selectedAttack,
    attackParameters = {},
) {
    // Keep the current choice while the catalog/model is still loading.
    if (!activeModel || !attackCatalog?.length) {
        return { attackId: selectedAttack, parameters: attackParameters }
    }

    const compatibleAttacks = getSupportedAttacksForTask(
        attackCatalog,
        activeModel.task_type || 'classification',
        activeModel.id,
    )
    const attackId = compatibleAttacks.some((attack) => attack.id === selectedAttack)
        ? selectedAttack
        : (compatibleAttacks.find((attack) => attack.id === 'random_noise')
            ?? compatibleAttacks[0])?.id ?? null

    return {
        attackId,
        parameters: attackId
            ? buildAttackParameters(attackCatalog, attackId, attackParameters)
            : {},
    }
}

export function buildAttackParameters(
    attackCatalog,
    attackId,
    currentParameters = {},
) {
    const attack = getAttackDefinition(attackCatalog, attackId)
    if (!attack) return { ...currentParameters }

    return Object.fromEntries(
        attack.parameters.map((parameter) => [
            parameter.key,
            Object.hasOwn(currentParameters, parameter.key)
                ? currentParameters[parameter.key]
                : parameter.default,
        ]),
    )
}

export function attackParametersKey(parameters = {}) {
    return JSON.stringify(
        Object.fromEntries(
            Object.entries(parameters).sort(([left], [right]) =>
                left.localeCompare(right),
            ),
        ),
    )
}
