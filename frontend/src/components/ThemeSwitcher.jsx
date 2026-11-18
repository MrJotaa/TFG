import { useTheme } from '../context/ThemeContext'

const THEME_OPTIONS = [
    { value: 'system', label: 'Sistema', icon: 'system' },
    { value: 'light', label: 'Claro', icon: 'sun' },
    { value: 'dark', label: 'Oscuro', icon: 'moon' },
]

function ThemeIcon({ name }) {
    if (name === 'sun') {
        return (
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="3.5" />
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
            </svg>
        )
    }

    if (name === 'moon') {
        return (
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M20.25 15.2A8.5 8.5 0 0 1 8.8 3.75 8.5 8.5 0 1 0 20.25 15.2Z" />
            </svg>
        )
    }

    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="4" width="18" height="12" rx="2" />
            <path d="M8 20h8M12 16v4" />
        </svg>
    )
}

function ThemeSwitcher({ compact = false, className = '' }) {
    const { themePreference, setThemePreference } = useTheme()

    const compactClassName = compact ? ' compact' : ''
    const additionalClassName = className ? ` ${className}` : ''
    const switcherClassName = `theme-switcher${compactClassName}${additionalClassName}`

    return (
        <fieldset className={`${switcherClassName} semantic-fieldset-reset`} aria-label="Tema de la aplicación">
            {THEME_OPTIONS.map((option) => (
                <button
                    key={option.value}
                    type="button"
                    className="theme-option"
                    aria-pressed={themePreference === option.value}
                    aria-label={`Usar tema ${option.label.toLowerCase()}`}
                    title={`Tema: ${option.label}`}
                    onClick={() => setThemePreference(option.value)}
                >
                    <ThemeIcon name={option.icon} />
                    <span>{option.label}</span>
                </button>
            ))}
        </fieldset>
    )
}

export default ThemeSwitcher
