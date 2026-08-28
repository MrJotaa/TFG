export const PASSWORD_MISMATCH_MESSAGE = 'Las contraseñas no coinciden.'

export function passwordsMatch(password, confirmation) {
    return password === confirmation
}

export function getPasswordConfirmationError(password, confirmation, touched = false) {
    if (!touched && confirmation.length === 0) return ''
    return passwordsMatch(password, confirmation) ? '' : PASSWORD_MISMATCH_MESSAGE
}
