import { Link, useNavigate } from 'react-router-dom'
import UserSelector from '../components/UserSelector'
import { useExperiment } from '../context/ExperimentContext'

function SignupPage() {
    const navigate = useNavigate()
    const { selectedUser, setSelectedUser } = useExperiment()

    function handleContinue() {
        if (selectedUser) {
            navigate('/dashboard')
        }
    }

    return (
        <main className="auth-page">
            <section className="auth-card">
                <h1>Adversarial AI App</h1>
                <h2>Crear usuario</h2>

                <p>
                    Crea un usuario para asociar datasets, imágenes y resultados.
                </p>

                <UserSelector
                    selectedUser={selectedUser}
                    onUserSelect={setSelectedUser}
                    showExistingUsers={false}
                    showCreateForm
                    onUserCreated={handleContinue}
                />

                {selectedUser && (
                    <button
                        type="button"
                        className="primary-action auth-submit"
                        onClick={() => navigate('/dashboard')}
                    >
                        Ir al panel principal
                    </button>
                )}

                <p>
                    ¿Ya tienes usuario? <Link to="/">Iniciar sesión</Link>
                </p>
            </section>
        </main>
    )
}

export default SignupPage