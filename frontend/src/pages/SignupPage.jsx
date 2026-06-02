import { Link, useNavigate } from 'react-router-dom'

function SignupPage() {
    const navigate = useNavigate()

    function handleMockSignup(event) {
        event.preventDefault()
        navigate('/dashboard')
    }

    return (
        <main className="auth-page">
            <section className="auth-card">
                <h1>Adversarial AI App</h1>
                <h2>Sign up</h2>

                <form onSubmit={handleMockSignup}>
                    <label>
                        Username
                        <input type="text" placeholder="Username" />
                    </label>

                    <label>
                        Email
                        <input type="email" placeholder="user@example.com" />
                    </label>

                    <label>
                        Password
                        <input type="password" placeholder="Password" />
                    </label>

                    <button type="submit" className="primary-action">
                        Crear cuenta
                    </button>
                </form>

                <p>
                    ¿Ya tienes cuenta? <Link to="/">Iniciar sesión</Link>
                </p>
            </section>
        </main>
    )
}

export default SignupPage