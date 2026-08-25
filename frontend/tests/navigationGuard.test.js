import assert from 'node:assert/strict'
import test from 'node:test'

import {
    getNavigationGuardDialogCopy,
    protectBeforeUnload,
    shouldProtectNavigation,
} from '../src/utils/navigationGuard.js'

test('solo protege la navegación cuando existe trabajo temporal o una operación en curso', () => {
    assert.equal(shouldProtectNavigation(), false)
    assert.equal(shouldProtectNavigation({ hasTemporaryWork: false, busy: false }), false)
    assert.equal(shouldProtectNavigation({ hasTemporaryWork: true, busy: false }), true)
    assert.equal(shouldProtectNavigation({ hasTemporaryWork: false, busy: true }), true)
})

test('distingue resultados temporales de operaciones en curso', () => {
    const temporaryCopy = getNavigationGuardDialogCopy({
        hasTemporaryWork: true,
        busy: false,
    })
    const busyCopy = getNavigationGuardDialogCopy({ busy: true })

    assert.equal(temporaryCopy.title, 'Resultados temporales sin guardar')
    assert.match(temporaryCopy.message, /se perderán/i)
    assert.equal(temporaryCopy.confirmLabel, 'Salir y descartar')

    assert.equal(busyCopy.title, 'Análisis en curso')
    assert.match(busyCopy.message, /se cancelará/i)
    assert.equal(busyCopy.confirmLabel, 'Salir y cancelar análisis')
})

test('cerrar sesión utiliza una confirmación específica', () => {
    const copy = getNavigationGuardDialogCopy({
        busy: true,
        intent: 'logout',
    })

    assert.equal(copy.title, 'Cerrar sesión')
    assert.equal(copy.confirmLabel, 'Cerrar sesión')
    assert.equal(copy.cancelLabel, 'Seguir analizando')
    assert.equal(copy.tone, 'danger')
})

test('beforeunload marca el evento para que el navegador solicite confirmación', () => {
    let prevented = false
    const event = {
        returnValue: null,
        preventDefault: () => {
            prevented = true
        },
    }

    assert.equal(protectBeforeUnload(event), true)
    assert.equal(prevented, true)
    assert.equal(event.returnValue, '')
    assert.equal(protectBeforeUnload(null), false)
})
