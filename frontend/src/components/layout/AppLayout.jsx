import { Outlet, useMatch, useNavigate } from 'react-router-dom'
import ThemeSwitcher from '../ThemeSwitcher'
import { useAuth } from '../../context/AuthContext'
import { useExperiment } from '../../context/ExperimentContext'
import { useNavigationGuard } from '../../context/NavigationGuardContext'

function BrandMark() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3 4.8 6.2v5.2c0 4.5 2.8 7.8 7.2 9.6 4.4-1.8 7.2-5.1 7.2-9.6V6.2L12 3Z" />
            <path d="m8.4 12.2 2.1 2.1 5.1-5.1" />
        </svg>
    )
}

function HomeIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m3 10 9-7 9 7" />
            <path d="M5 9v11h5v-6h4v6h5V9" />
        </svg>
    )
}

function LogoutIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M10 5H6.8A1.8 1.8 0 0 0 5 6.8v10.4A1.8 1.8 0 0 0 6.8 19H10" />
            <path d="M14.5 8.5 18 12l-3.5 3.5M9 12h9" />
        </svg>
    )
}

function AppLayout() {
    const navigate = useNavigate()
    const isDashboard = Boolean(useMatch({ path: '/dashboard', end: true }))
    const { logout, user } = useAuth()
    const { resetExperiment } = useExperiment()
    const { requestNavigation } = useNavigationGuard()

    function navigateSafely(path) {
        requestNavigation(() => navigate(path))
    }

    function handleLogout() {
        requestNavigation(
            () => {
                logout()
                resetExperiment()
                navigate('/')
            },
            { intent: 'logout' },
        )
    }

    const userInitial = user?.name?.trim()?.charAt(0)?.toUpperCase() || 'U'

    return (
        <main className="app-shell">
            <header
                className="app-header app-header-with-actions app-session-header"
                aria-label="Navegación principal y sesión de usuario"
            >
                <button
                    type="button"
                    className="app-brand"
                    onClick={() => navigateSafely('/dashboard')}
                    aria-label="Ir al panel principal"
                    title="Volver al panel principal"
                >
                    <span className="app-brand-mark">
                        {isDashboard ? <BrandMark /> : <HomeIcon />}
                    </span>
                    <span className="app-brand-copy">
                        <strong>Analizador adversario</strong>
                        <small>Evaluación de robustez</small>
                    </span>
                </button>

                <div className="header-user-actions">
                    <ThemeSwitcher compact className="header-theme-switcher" />

                    {user && (
                        <button
                            type="button"
                            className="header-user-link"
                            onClick={() => navigateSafely('/profile')}
                            title="Abrir configuración de usuario"
                        >
                            <span className="header-user-avatar" aria-hidden="true">
                                {userInitial}
                            </span>
                            <span className="header-user-copy">
                                <small>Usuario activo</small>
                                <strong>{user.name}</strong>
                            </span>
                        </button>
                    )}

                    <button
                        type="button"
                        className="header-logout-button"
                        onClick={handleLogout}
                        title="Cerrar sesión"
                    >
                        <LogoutIcon />
                        <span>Cerrar sesión</span>
                    </button>
                </div>
            </header>

            <Outlet />

        </main>
    )
}

export default AppLayout
