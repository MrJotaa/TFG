export const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
    yolo_confidence: 'YOLO Confidence',
}

export function getAttackName(attackId) {
    return ATTACK_NAMES[attackId] || attackId
}

export function formatAttackParameters(selectedAttack, attackParameters) {
    const baseParameters = `ε=${attackParameters.epsilon}`

    const attacksWithIterations = ['pgd', 'yolo_confidence']

    if (attacksWithIterations.includes(selectedAttack)) {
        return `${baseParameters} · α=${attackParameters.alpha} · steps=${attackParameters.steps}`
    }

    return baseParameters
}