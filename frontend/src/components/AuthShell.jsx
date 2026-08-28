import ThemeSwitcher from './ThemeSwitcher'

function BrandMark() {
    return (
        <span className="auth-brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24">
                <path d="M12 3 5.5 5.7v5.7c0 4.2 2.6 7.6 6.5 9.6 3.9-2 6.5-5.4 6.5-9.6V5.7L12 3Z" />
                <path d="m9.1 12 1.8 1.8 4-4.1" />
            </svg>
        </span>
    )
}

function AuthShell({ eyebrow, title, description, children }) {
    return (
        <main className="auth-page">
            <header className="auth-topbar">
                <div className="auth-brand">
                    <BrandMark />
                    <div>
                        <strong>Analizador adversario</strong>
                        <span>Evaluación de robustez</span>
                    </div>
                </div>
                <ThemeSwitcher compact />
            </header>

            <section className="auth-layout">
                <section className="auth-form-panel">
                    <div className="auth-form-heading">
                        <span className="auth-form-eyebrow">{eyebrow}</span>
                        <h1>{title}</h1>
                        <p>{description}</p>
                    </div>
                    {children}
                </section>
            </section>
        </main>
    )
}

export default AuthShell
