import assert from 'node:assert/strict'
import test from 'node:test'

import {
    THEME_STORAGE_KEY,
    applyThemePreference,
    getStoredThemePreference,
    getSystemTheme,
    normalizeThemePreference,
    persistThemePreference,
    resolveTheme,
} from '../src/utils/theme.js'

test('normaliza preferencias desconocidas al tema del sistema', () => {
    assert.equal(normalizeThemePreference('dark'), 'dark')
    assert.equal(normalizeThemePreference('light'), 'light')
    assert.equal(normalizeThemePreference('system'), 'system')
    assert.equal(normalizeThemePreference('sepia'), 'system')
    assert.equal(normalizeThemePreference(null), 'system')
})

test('resuelve la preferencia system contra prefers-color-scheme', () => {
    const darkMedia = () => ({ matches: true })
    const lightMedia = () => ({ matches: false })

    assert.equal(getSystemTheme(darkMedia), 'dark')
    assert.equal(getSystemTheme(lightMedia), 'light')
    assert.equal(resolveTheme('system', darkMedia), 'dark')
    assert.equal(resolveTheme('system', lightMedia), 'light')
    assert.equal(resolveTheme('dark', lightMedia), 'dark')
})

test('persiste y recupera la preferencia de tema', () => {
    const values = new Map()
    const storage = {
        getItem: (key) => values.get(key) ?? null,
        setItem: (key, value) => values.set(key, value),
    }

    persistThemePreference('dark', storage)

    assert.equal(values.get(THEME_STORAGE_KEY), 'dark')
    assert.equal(getStoredThemePreference(storage), 'dark')
})

test('normaliza una preferencia inválida antes de persistirla', () => {
    const values = new Map()
    const storage = {
        getItem: (key) => values.get(key) ?? null,
        setItem: (key, value) => values.set(key, value),
    }

    assert.equal(persistThemePreference('sepia', storage), 'system')
    assert.equal(values.get(THEME_STORAGE_KEY), 'system')
})

test('tolera almacenamiento no disponible', () => {
    const brokenStorage = {
        getItem: () => {
            throw new Error('storage blocked')
        },
        setItem: () => {
            throw new Error('storage blocked')
        },
    }

    assert.equal(getStoredThemePreference(brokenStorage), 'system')
    assert.equal(persistThemePreference('dark', brokenStorage), 'dark')
})

test('aplica tema resuelto y preferencia al elemento raíz', () => {
    const root = { dataset: {}, style: {} }
    const darkMedia = () => ({ matches: true })

    const resolved = applyThemePreference('system', root, darkMedia)

    assert.equal(resolved, 'dark')
    assert.equal(root.dataset.theme, 'dark')
    assert.equal(root.dataset.themePreference, 'system')
    assert.equal(root.style.colorScheme, 'dark')
})
