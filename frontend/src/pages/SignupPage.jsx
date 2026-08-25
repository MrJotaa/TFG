import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthShell from '../components/AuthShell'
import PasswordField from '../components/PasswordField'
import { useAuth } from '../context/AuthContext'
import {
    getPasswordConfirmationError,
    PASSWORD_MISMATCH_MESSAGE,
    passwordsMatch,
} from '../utils/authForms'

function SignupPage() {
    const navigate = useNavigate()
    const { signup } = useAuth()

    const [name, setName] = useState('')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [confirmTouched, setConfirmTouched] = useState(false)
    const [showPasswords, setShowPasswords] = useState(false)
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const confirmationError = getPasswordConfirmationError(
        password,
        confirmPassword,
        confirmTouched,
    )

    async function handleSignup(event) {
        event.preventDefault()

        if (!passwordsMatch(password, confirmPassword)) {
            setConfirmTouched(true)
            setError('')
            return
        }

        try {
            setLoading(true)
            setError('')

            await signup({ name, email, password })
            navigate('/dashboard')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <AuthShell
            eyebrow="Registro"
            title="Crea tu espacio de análisis"
            description="Configura tu cuenta y empieza a evaluar la robustez de tus modelos."
        >
            {error && (
                <div className="auth-alert" role="alert">
                    <span className="auth-alert-icon" aria-hidden="true">!</span>
                    <span>{error}</span>
                </div>
            )}

            <form className="auth-form" onSubmit={handleSignup} aria-busy={loading}>
                <label className="auth-field" htmlFor="signup-name">
                    <span className="auth-field-label">Nombre</span>
                    <input
                        id="signup-name"
                        type="text"
                        autoComplete="name"
                        placeholder="Tu nombre"
                        value={name}
                        onChange={(event) => {
                            setName(event.target.value)
                            if (error) setError('')
                        }}
                        required
                        disabled={loading}
                    />
                </label>

                <label className="auth-field" htmlFor="signup-email">
                    <span className="auth-field-label">Correo electrónico</span>
                    <input
                        id="signup-email"
                        type="email"
                        inputMode="email"
                        autoComplete="email"
                        spellCheck="false"
                        placeholder="usuario@email.com"
                        value={email}
                        onChange={(event) => {
                            setEmail(event.target.value)
                            if (error) setError('')
                        }}
                        required
                        disabled={loading}
                    />
                </label>

                <PasswordField
                    id="signup-password"
                    label="Contraseña"
                    value={password}
                    onChange={(event) => {
                        setPassword(event.target.value)
                        if (error) setError('')
                    }}
                    autoComplete="new-password"
                    disabled={loading}
                    visible={showPasswords}
                    onVisibleChange={setShowPasswords}
                    visibilityLabel="contraseñas"
                />

                <PasswordField
                    id="signup-password-confirmation"
                    label="Confirmar contraseña"
                    value={confirmPassword}
                    onChange={(event) => {
                        setConfirmPassword(event.target.value)
                        if (error) setError('')
                    }}
                    onBlur={() => setConfirmTouched(true)}
                    autoComplete="new-password"
                    placeholder="Repite tu contraseña"
                    error={confirmationError}
                    disabled={loading}
                    visible={showPasswords}
                    showToggle={false}
                />

                <button
                    type="submit"
                    className="primary-action auth-submit"
                    disabled={loading || Boolean(confirmationError)}
                    title={confirmationError ? PASSWORD_MISMATCH_MESSAGE : undefined}
                >
                    {loading && <span className="auth-submit-spinner" aria-hidden="true" />}
                    <span>{loading ? 'Creando cuenta…' : 'Crear cuenta'}</span>
                </button>
            </form>

            <p className="auth-alternative">
                ¿Ya tienes una cuenta? <Link to="/">Iniciar sesión</Link>
            </p>
        </AuthShell>
    )
}

export default SignupPage
