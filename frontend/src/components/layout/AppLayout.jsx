import { NavLink, Outlet } from 'react-router-dom'

function AppLayout() {
    return (
        <main className="app-shell">
            <header className="app-header">
                <h1>Adversarial AI App</h1>

                <nav className="app-nav">
                    <NavLink to="/dashboard">Dashboard</NavLink>
                    <NavLink to="/datasets">Datasets</NavLink>
                    <NavLink to="/prediction">Predicción</NavLink>
                    <NavLink to="/comparison">Comparación</NavLink>
                    <NavLink to="/evaluation">Evaluación</NavLink>
                </nav>
            </header>

            <Outlet />

            <footer className="app-footer">
                © 2026 Adversarial AI App. All rights reserved.
            </footer>
        </main>
    )
}

export default AppLayout