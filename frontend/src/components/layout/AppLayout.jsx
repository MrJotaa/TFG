import { Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'
import { useExperiment } from '../../context/ExperimentContext'

function AppLayout() {
    const navigate = useNavigate()
    const { logout, user } = useAuth()
    const { resetExperiment } = useExperiment()

    function handleLogout() {
        logout()
        resetExperiment()
        navigate('/')
    }

    return (
        <main className="app-shell">
            <header className="app-header app-header-with-actions">
                <h1>Adversarial AI App</h1>

                <div className="header-user-actions">
                    {user && (
                        <button
                            type="button"
                            className="header-user-link"
                            onClick={() => navigate('/profile')}
                        >
                            {user.name}
                        </button>
                    )}

                    <button
                        type="button"
                        className="secondary-action small"
                        onClick={handleLogout}
                    >
                        Cerrar sesión
                    </button>
                </div>
            </header>

            <Outlet />

            <footer className="app-footer">
                © 2026 Adversarial AI App. All rights reserved.
            </footer>
        </main>
    )
}

export default AppLayout