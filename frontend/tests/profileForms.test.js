import assert from 'node:assert/strict'
import test from 'node:test'

import {
    buildProfileUpdatePayload,
    CURRENT_PASSWORD_REQUIRED_MESSAGE,
    getProfilePasswordErrors,
    hasPasswordErrors,
    hasProfileChanges,
    PROFILE_PASSWORD_MISMATCH_MESSAGE,
    wantsPasswordChange,
} from '../src/utils/profileForms.js'

const user = {
    id: 'user-1',
    name: 'Javi',
    email: 'javi@example.com',
}

test('no considera un cambio de contraseña cuando los tres campos están vacíos', () => {
    assert.equal(wantsPasswordChange({}), false)
})

test('exige la contraseña actual cuando se intenta establecer una nueva', () => {
    const errors = getProfilePasswordErrors({
        newPassword: 'nueva-segura',
        confirmPassword: 'nueva-segura',
        attempted: true,
    })

    assert.equal(errors.currentPassword, CURRENT_PASSWORD_REQUIRED_MESSAGE)
    assert.equal(hasPasswordErrors(errors), true)
})

test('detecta una confirmación de contraseña distinta', () => {
    const errors = getProfilePasswordErrors({
        currentPassword: 'actual',
        newPassword: 'nueva-segura',
        confirmPassword: 'otra',
        confirmationTouched: true,
    })

    assert.equal(errors.confirmPassword, PROFILE_PASSWORD_MISMATCH_MESSAGE)
})

test('considera válido el cambio cuando las credenciales están completas y coinciden', () => {
    const errors = getProfilePasswordErrors({
        currentPassword: 'actual',
        newPassword: 'nueva-segura',
        confirmPassword: 'nueva-segura',
        attempted: true,
    })

    assert.equal(hasPasswordErrors(errors), false)
})

test('detecta cambios de identidad y de contraseña', () => {
    assert.equal(
        hasProfileChanges({
            user,
            name: 'Javi',
            email: 'javi@example.com',
            passwordFields: {},
        }),
        false,
    )
    assert.equal(
        hasProfileChanges({
            user,
            name: 'Javier',
            email: 'javi@example.com',
            passwordFields: {},
        }),
        true,
    )
    assert.equal(
        hasProfileChanges({
            user,
            name: 'Javi',
            email: 'javi@example.com',
            passwordFields: { newPassword: 'nueva' },
        }),
        true,
    )
})

test('solo incluye la contraseña actual cuando existe un cambio de contraseña válido', () => {
    assert.deepEqual(
        buildProfileUpdatePayload({
            user,
            name: 'Javier',
            email: 'javi@example.com',
            currentPassword: '',
            newPassword: '',
            confirmPassword: '',
        }),
        { name: 'Javier' },
    )

    assert.deepEqual(
        buildProfileUpdatePayload({
            user,
            name: 'Javi',
            email: 'javi@example.com',
            currentPassword: 'actual',
            newPassword: 'nueva-segura',
            confirmPassword: 'nueva-segura',
        }),
        {
            password: 'nueva-segura',
            current_password: 'actual',
        },
    )
})
