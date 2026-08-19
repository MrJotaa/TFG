import { getAttackDefinition } from './attackCatalog.js'

export function getAttackName(attackId, attackCatalog = []) {
    return getAttackDefinition(attackCatalog, attackId)?.name || attackId
}

export function formatAttackParameters(
    selectedAttack,
    attackParameters,
    attackCatalog = [],
) {
    const attack = getAttackDefinition(attackCatalog, selectedAttack)

    if (!attack) {
        const entries = Object.entries(attackParameters ?? {})
        if (entries.length === 0) return 'Sin parámetros'
        return entries
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([key, value]) => `${key}=${value}`)
            .join(' · ')
    }

    if (attack.parameters.length === 0) return 'Sin parámetros'

    return attack.parameters
        .map((parameter) => {
            const value = Object.hasOwn(attackParameters ?? {}, parameter.key)
                ? attackParameters[parameter.key]
                : parameter.default
            return `${parameter.summary_label}=${value}`
        })
        .join(' · ')
}
