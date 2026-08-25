import { useEffect } from 'react'

function CloseIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m6 6 12 12M18 6 6 18" />
        </svg>
    )
}

function DatasetDialog({ title, eyebrow, children, onClose, busy = false, danger = false }) {
    useEffect(() => {
        function handleKeyDown(event) {
            if (event.key === 'Escape' && !busy) onClose()
        }
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [busy, onClose])

    return (
        <div
            className="dataset-dialog-backdrop"
            role="presentation"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget && !busy) onClose()
            }}
        >
            <section
                className={`dataset-dialog${danger ? ' danger' : ''}`}
                role="dialog"
                aria-modal="true"
                aria-label={title}
            >
                <header className="dataset-dialog-header">
                    <div>
                        {eyebrow && <span>{eyebrow}</span>}
                        <h2>{title}</h2>
                    </div>
                    <button
                        type="button"
                        className="dataset-dialog-close"
                        onClick={onClose}
                        disabled={busy}
                        aria-label="Cerrar"
                    >
                        <CloseIcon />
                    </button>
                </header>
                {children}
            </section>
        </div>
    )
}

export default DatasetDialog
