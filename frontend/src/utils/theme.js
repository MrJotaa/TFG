export const THEME_STORAGE_KEY = 'tfg-theme'

export const THEME_PREFERENCES = Object.freeze(['system', 'light', 'dark'])

function getBrowserStorage() {
    try {
        return globalThis.localStorage ?? null
    } catch {
        return null
    }
}

function getBrowserMatchMedia() {
    if (typeof globalThis.matchMedia !== 'function') {
        return undefined
    }

    return globalThis.matchMedia.bind(globalThis)
}

export function normalizeThemePreference(value) {
    return THEME_PREFERENCES.includes(value) ? value : 'system'
}

export function getStoredThemePreference(storage = getBrowserStorage()) {
    if (!storage) {
        return 'system'
    }

    try {
        return normalizeThemePreference(storage.getItem(THEME_STORAGE_KEY))
    } catch {
        return 'system'
    }
}

export function getSystemTheme(matchMedia = getBrowserMatchMedia()) {
    if (typeof matchMedia !== 'function') {
        return 'light'
    }

    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function resolveTheme(preference, matchMedia = getBrowserMatchMedia()) {
    const normalizedPreference = normalizeThemePreference(preference)
    return normalizedPreference === 'system'
        ? getSystemTheme(matchMedia)
        : normalizedPreference
}

export function applyThemePreference(
    preference,
    root = globalThis.document?.documentElement,
    matchMedia = getBrowserMatchMedia(),
) {
    const normalizedPreference = normalizeThemePreference(preference)
    const resolvedTheme = resolveTheme(normalizedPreference, matchMedia)

    if (root) {
        root.dataset.theme = resolvedTheme
        root.dataset.themePreference = normalizedPreference
        root.style.colorScheme = resolvedTheme
    }

    return resolvedTheme
}

export function persistThemePreference(
    preference,
    storage = getBrowserStorage(),
) {
    const normalizedPreference = normalizeThemePreference(preference)

    if (storage) {
        try {
            storage.setItem(THEME_STORAGE_KEY, normalizedPreference)
        } catch {
            // localStorage can be unavailable in restricted browser contexts.
        }
    }

    return normalizedPreference
}

export function initializeTheme() {
    const preference = getStoredThemePreference()
    return {
        preference,
        resolvedTheme: applyThemePreference(preference),
    }
}
