import { passwordsMatch } from './authForms.js'

export const CURRENT_PASSWORD_REQUIRED_MESSAGE =
    'Introduce tu contraseña actual para cambiarla.'
export const NEW_PASSWORD_REQUIRED_MESSAGE = 'Introduce una contraseña nueva.'
export const CONFIRM_PASSWORD_REQUIRED_MESSAGE = 'Confirma la nueva contraseña.'
export const PROFILE_PASSWORD_MISMATCH_MESSAGE = 'Las contraseñas no coinciden.'

export function wantsPasswordChange({
    currentPassword = '',
    newPassword = '',
    confirmPassword = '',
}) {
    return Boolean(currentPassword || newPassword || confirmPassword)
}

export function getProfilePasswordErrors({
    currentPassword = '',
    newPassword = '',
    confirmPassword = '',
    attempted = false,
    confirmationTouched = false,
}) {
    const wantsChange = wantsPasswordChange({
        currentPassword,
        newPassword,
        confirmPassword,
    })

    if (!wantsChange) {
        return {
            currentPassword: '',
            newPassword: '',
            confirmPassword: '',
        }
    }

    let confirmPasswordError = ''
    if ((attempted || confirmationTouched) && !confirmPassword) {
        confirmPasswordError = CONFIRM_PASSWORD_REQUIRED_MESSAGE
    } else if (confirmPassword && !passwordsMatch(newPassword, confirmPassword)) {
        confirmPasswordError = PROFILE_PASSWORD_MISMATCH_MESSAGE
    }

    return {
        currentPassword: attempted && !currentPassword ? CURRENT_PASSWORD_REQUIRED_MESSAGE : '',
        newPassword: attempted && !newPassword ? NEW_PASSWORD_REQUIRED_MESSAGE : '',
        confirmPassword: confirmPasswordError,
    }
}

export function hasPasswordErrors(errors) {
    return Boolean(
        errors.currentPassword || errors.newPassword || errors.confirmPassword,
    )
}

export function hasProfileChanges({ user, name, email, passwordFields }) {
    if (!user) return false

    const identityChanged =
        name.trim() !== user.name || email.trim() !== user.email

    return identityChanged || wantsPasswordChange(passwordFields)
}

export function buildProfileUpdatePayload({
    user,
    name,
    email,
    currentPassword,
    newPassword,
    confirmPassword,
}) {
    const payload = {}
    const normalizedName = name.trim()
    const normalizedEmail = email.trim()

    if (normalizedName !== user.name) payload.name = normalizedName
    if (normalizedEmail !== user.email) payload.email = normalizedEmail

    if (
        wantsPasswordChange({ currentPassword, newPassword, confirmPassword }) &&
        currentPassword &&
        newPassword &&
        passwordsMatch(newPassword, confirmPassword)
    ) {
        payload.password = newPassword
        payload.current_password = currentPassword
    }

    return payload
}
