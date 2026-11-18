import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AuthShell from '../components/AuthShell'
import PasswordField from '../components/PasswordField'
import { useAuth } from '../context/AuthContext'

function LoginPage() {
    const navigate = useNavigate()
    const { login } = useAuth()

    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    async function handleLogin(event) {
        event.preventDefault()

        try {
            setLoading(true)
            setError('')

            await login({ email, password })
            navigate('/dashboard')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <AuthShell
            eyebrow="Acceso"
            title="Bienvenido de nuevo"
            description="Accede a tu espacio de experimentación y continúa tus análisis."
        >
            {error && (
                <div className="auth-alert" role="alert">
                    <span className="auth-alert-icon" aria-hidden="true">!</span>
                    <span>{error}</span>
                </div>
            )}

            <form className="auth-form" onSubmit={handleLogin} aria-busy={loading}>
                <label className="auth-field" htmlFor="login-email">
                    <span className="auth-field-label">Correo electrónico</span>
                    <input
                        id="login-email"
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
                    id="login-password"
                    label="Contraseña"
                    value={password}
                    onChange={(event) => {
                        setPassword(event.target.value)
                        if (error) setError('')
                    }}
                    autoComplete="current-password"
                    disabled={loading}
                />

                <button
                    type="submit"
                    className="primary-action auth-submit"
                    disabled={loading}
                >
                    {loading && <span className="auth-submit-spinner" aria-hidden="true" />}
                    <span>{loading ? 'Iniciando sesión…' : 'Iniciar sesión'}</span>
                </button>
            </form>

            <p className="auth-alternative">
                ¿No tienes una cuenta? <Link to="/signup">Crear cuenta</Link>
            </p>
        </AuthShell>
    )
}

export default LoginPage
