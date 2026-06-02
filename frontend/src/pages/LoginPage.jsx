import { Link, useNavigate } from 'react-router-dom'

function LoginPage() {
    const navigate = useNavigate()

    function handleMockLogin(event) {
        event.preventDefault()
        navigate('/dashboard')
    }

    return (
        <main className="auth-page">
            <section className="auth-card">
                <h1>Adversarial AI App</h1>
                <h2>Login</h2>

                <form onSubmit={handleMockLogin}>
                    <label>
                        Email
                        <input type="email" placeholder="user@example.com" />
                    </label>

                    <label>
                        Password
                        <input type="password" placeholder="Password" />
                    </label>

                    <button type="submit" className="primary-action">
                        Entrar
                    </button>
                </form>

                <p>
                    ¿No tienes cuenta? <Link to="/signup">Crear cuenta</Link>
                </p>
            </section>
        </main>
    )
}

export default LoginPage