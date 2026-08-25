import { Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
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
        <main className="auth-page">
            <section className="auth-card">
                <h1>Analizador de ataques adversarios</h1>
                <h2>Iniciar sesión</h2>

                <p>Introduce tus credenciales para acceder al panel principal.</p>

                {error && <p className="error">{error}</p>}

                <form onSubmit={handleLogin}>
                    <label>
                        Correo
                        <input
                            type="email"
                            placeholder="usuario@email.com"
                            value={email}
                            onChange={(event) => setEmail(event.target.value)}
                            required
                        />
                    </label>

                    <label>
                        Contraseña
                        <input
                            type="password"
                            placeholder="Contraseña"
                            value={password}
                            onChange={(event) => setPassword(event.target.value)}
                            required
                        />
                    </label>

                    <button
                        type="submit"
                        className="primary-action auth-submit"
                        disabled={loading}
                    >
                        {loading ? 'Entrando...' : 'Entrar'}
                    </button>
                </form>

                <p>
                    ¿No tienes usuario? <Link to="/signup">Crear usuario</Link>
                </p>
            </section>
        </main>
    )
}

export default LoginPage