import { Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { useAuth } from '../context/AuthContext'

function SignupPage() {
    const navigate = useNavigate()
    const { signup } = useAuth()

    const [name, setName] = useState('')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    async function handleSignup(event) {
        event.preventDefault()

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
        <main className="auth-page">
            <section className="auth-card">
                <h1>Analizador de ataques adversarios</h1>
                <h2>Crear usuario</h2>

                <p>Crea una cuenta para gestionar conjuntos de datos y experimentos.</p>

                {error && <p className="error">{error}</p>}

                <form onSubmit={handleSignup}>
                    <label>
                        Nombre
                        <input
                            type="text"
                            placeholder="Nombre"
                            value={name}
                            onChange={(event) => setName(event.target.value)}
                            required
                        />
                    </label>

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
                        {loading ? 'Creando usuario...' : 'Crear usuario'}
                    </button>
                </form>

                <p>
                    ¿Ya tienes usuario? <Link to="/">Iniciar sesión</Link>
                </p>
            </section>
        </main>
    )
}

export default SignupPage