export const ATTACK_CATALOG = Object.freeze([
    Object.freeze({
        id: 'fgsm',
        name: 'FGSM',
        description: 'Ataque rápido basado en gradiente',
        supportedTaskTypes: Object.freeze(['classification']),
    }),
    Object.freeze({
        id: 'random_noise',
        name: 'Random Noise',
        description: 'Baseline con ruido aleatorio',
        supportedTaskTypes: Object.freeze(['classification', 'detection']),
    }),
    Object.freeze({
        id: 'pgd',
        name: 'PGD',
        description: 'Ataque iterativo basado en gradiente',
        supportedTaskTypes: Object.freeze(['classification']),
    }),
    Object.freeze({
        id: 'yolo_confidence',
        name: 'YOLO Confidence',
        description: 'Ataque de supresión de confianza para detección',
        supportedTaskTypes: Object.freeze(['detection']),
    }),
])

export function isAttackSupportedForTask(attackId, taskType) {
    const attack = ATTACK_CATALOG.find((candidate) => candidate.id === attackId)
    return Boolean(attack?.supportedTaskTypes.includes(taskType))
}

export function getSupportedAttacksForTask(taskType) {
    return ATTACK_CATALOG.filter((attack) =>
        attack.supportedTaskTypes.includes(taskType),
    )
}
