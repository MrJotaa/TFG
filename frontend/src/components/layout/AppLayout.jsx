import { Outlet } from 'react-router-dom'

function AppLayout() {
    return (
        <main className="app-shell">
            <header className="app-header">
                <h1>Adversarial AI App</h1>
            </header>

            <Outlet />

            <footer className="app-footer">
                © 2026 Adversarial AI App. All rights reserved.
            </footer>
        </main>
    )
}

export default AppLayout