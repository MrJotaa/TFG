import { Link, useNavigate } from 'react-router-dom'
import UserSelector from '../components/UserSelector'
import { useExperiment } from '../context/ExperimentContext'

function LoginPage() {
    const navigate = useNavigate()
    const { selectedUser, setSelectedUser } = useExperiment()

    function handleContinue(event) {
        event.preventDefault()

        if (!selectedUser) {
            return
        }

        navigate('/dashboard')
    }

    return (
        <main className="auth-page">
            <section className="auth-card">
                <h1>Adversarial AI App</h1>
                <h2>Iniciar sesión</h2>

                <p>
                    Selecciona un usuario existente para acceder al panel principal.
                </p>

                <UserSelector
                    selectedUser={selectedUser}
                    onUserSelect={setSelectedUser}
                    showCreateForm={false}
                />

                <form onSubmit={handleContinue}>
                    <button
                        type="submit"
                        className="primary-action auth-submit"
                        disabled={!selectedUser}
                    >
                        Entrar
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