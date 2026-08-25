import { useState } from 'react'

function EyeIcon({ visible }) {
    if (visible) {
        return (
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M3 3l18 18" />
                <path d="M10.7 10.7a2 2 0 0 0 2.6 2.6" />
                <path d="M9.9 4.25A10.8 10.8 0 0 1 12 4c5.5 0 9 4.5 9 8a8.7 8.7 0 0 1-2 4.7" />
                <path d="M6.6 6.6C4.35 8 3 10.1 3 12c0 3.5 3.5 8 9 8a10 10 0 0 0 4.1-.85" />
            </svg>
        )
    }

    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 12s3.5-8 9-8 9 8 9 8-3.5 8-9 8-9-8-9-8Z" />
            <circle cx="12" cy="12" r="2.5" />
        </svg>
    )
}

function PasswordField({
    id,
    label,
    value,
    onChange,
    onBlur,
    autoComplete,
    placeholder = 'Introduce tu contraseña',
    error = '',
    disabled = false,
    required = true,
    visible,
    onVisibleChange,
    showToggle = true,
    visibilityLabel,
}) {
    const [internalVisible, setInternalVisible] = useState(false)
    const isControlled = typeof visible === 'boolean'
    const isVisible = isControlled ? visible : internalVisible
    const errorId = error ? `${id}-error` : undefined
    const toggleLabel = visibilityLabel || label.toLowerCase()

    function toggleVisibility() {
        const nextVisible = !isVisible
        if (!isControlled) setInternalVisible(nextVisible)
        onVisibleChange?.(nextVisible)
    }

    return (
        <label className={`auth-field${error ? ' has-error' : ''}`} htmlFor={id}>
            <span className="auth-field-label">{label}</span>
            <span className="auth-password-control">
                <input
                    id={id}
                    type={isVisible ? 'text' : 'password'}
                    value={value}
                    onChange={onChange}
                    onBlur={onBlur}
                    autoComplete={autoComplete}
                    placeholder={placeholder}
                    required={required}
                    disabled={disabled}
                    aria-invalid={Boolean(error)}
                    aria-describedby={errorId}
                />
                {showToggle && (
                    <button
                        type="button"
                        className="auth-password-toggle"
                        onClick={toggleVisibility}
                        aria-label={isVisible ? `Ocultar ${toggleLabel}` : `Mostrar ${toggleLabel}`}
                        aria-pressed={isVisible}
                        disabled={disabled}
                    >
                        <EyeIcon visible={isVisible} />
                    </button>
                )}
            </span>
            {error && (
                <span id={errorId} className="auth-field-error" role="alert">
                    {error}
                </span>
            )}
        </label>
    )
}

export default PasswordField
