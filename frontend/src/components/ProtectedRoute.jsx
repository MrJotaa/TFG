import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

function ProtectedRoute() {
    const { isAuthenticated, loadingAuth } = useAuth()

    if (loadingAuth) {
        return (
            <main className="auth-page">
                <section className="auth-card">
                    <h1>Adversarial AI App</h1>
                    <p>Cargando sesión...</p>
                </section>
            </main>
        )
    }

    if (!isAuthenticated) {
        return <Navigate to="/" replace />
    }

    return <Outlet />
}

export default ProtectedRoute