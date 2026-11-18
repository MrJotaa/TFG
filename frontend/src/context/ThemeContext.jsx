import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
} from 'react'
import {
    applyThemePreference,
    getStoredThemePreference,
    normalizeThemePreference,
    persistThemePreference,
    resolveTheme,
} from '../utils/theme'

const ThemeContext = createContext(null)

function subscribeToSystemTheme(mediaQuery, listener) {
    if (typeof mediaQuery.addEventListener === 'function') {
        mediaQuery.addEventListener('change', listener)
        return () => mediaQuery.removeEventListener('change', listener)
    }

    if (typeof mediaQuery.addListener === 'function') {
        mediaQuery.addListener(listener)
        return () => mediaQuery.removeListener(listener)
    }

    return undefined
}

export function ThemeProvider({ children }) {
    const [themePreference, setThemePreference] = useState(() =>
        getStoredThemePreference(),
    )
    const [resolvedTheme, setResolvedTheme] = useState(() =>
        resolveTheme(getStoredThemePreference()),
    )

    useEffect(() => {
        const syncTheme = () => {
            setResolvedTheme(applyThemePreference(themePreference))
        }

        syncTheme()

        if (
            themePreference !== 'system' ||
            typeof window.matchMedia !== 'function'
        ) {
            return undefined
        }

        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
        return subscribeToSystemTheme(mediaQuery, syncTheme)
    }, [themePreference])

    const updateThemePreference = useCallback((nextPreference) => {
        const normalizedPreference = normalizeThemePreference(nextPreference)
        persistThemePreference(normalizedPreference)
        setResolvedTheme(applyThemePreference(normalizedPreference))
        setThemePreference(normalizedPreference)
    }, [])

    const value = useMemo(
        () => ({ themePreference, resolvedTheme, setThemePreference: updateThemePreference }),
        [themePreference, resolvedTheme, updateThemePreference],
    )

    return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

// Context hooks intentionally live beside their provider.
// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
    const context = useContext(ThemeContext)

    if (!context) {
        throw new Error('useTheme debe utilizarse dentro de ThemeProvider')
    }

    return context
}
