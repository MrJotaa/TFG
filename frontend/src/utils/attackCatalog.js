export function getAttackDefinition(attackCatalog, attackId) {
    return (attackCatalog ?? []).find((attack) => attack.id === attackId) ?? null
}

export function isAttackSupportedForTask(attackCatalog, attackId, taskType) {
    const attack = getAttackDefinition(attackCatalog, attackId)
    return Boolean(attack?.supported_task_types?.includes(taskType))
}

export function getSupportedAttacksForTask(attackCatalog, taskType) {
    return (attackCatalog ?? []).filter((attack) =>
        attack.supported_task_types?.includes(taskType),
    )
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
