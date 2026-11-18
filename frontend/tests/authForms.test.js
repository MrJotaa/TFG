import assert from 'node:assert/strict'
import test from 'node:test'

import {
    getPasswordConfirmationError,
    PASSWORD_MISMATCH_MESSAGE,
    passwordsMatch,
} from '../src/utils/authForms.js'

test('considera válidas dos contraseñas idénticas', () => {
    assert.equal(passwordsMatch('segura-123', 'segura-123'), true)
})

test('detecta contraseñas diferentes', () => {
    assert.equal(passwordsMatch('segura-123', 'segura-124'), false)
})

test('no muestra error antes de interactuar con la confirmación vacía', () => {
    assert.equal(getPasswordConfirmationError('segura-123', '', false), '')
})

test('muestra error tras interactuar cuando la confirmación no coincide', () => {
    assert.equal(
        getPasswordConfirmationError('segura-123', 'segura-124', true),
        PASSWORD_MISMATCH_MESSAGE,
    )
})

test('retira el error cuando ambas contraseñas vuelven a coincidir', () => {
    assert.equal(getPasswordConfirmationError('segura-123', 'segura-123', true), '')
})
