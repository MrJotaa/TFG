import {
    evaluationProgressMessage,
    evaluationProgressPercent,
} from '../../utils/evaluationProgress'

function EvaluationProgress({ progress, title, pendingMessage, onCancel }) {
    if (!progress) {
        return (
            <section
                className="evaluation-loading-card"
                role="status"
                aria-live="polite"
            >
                <div className="evaluation-loader" aria-hidden="true" />
                <div>
                    <strong>{title}</strong>
                    <p>{pendingMessage}</p>
                </div>
                {onCancel && (
                    <button
                        type="button"
                        className="secondary-action evaluation-cancel-button"
                        onClick={onCancel}
                    >
                        Cancelar
                    </button>
                )}
            </section>
        )
    }

    const percent = evaluationProgressPercent(progress)
    const message = evaluationProgressMessage(progress)

    return (
        <section
            className="evaluation-loading-card evaluation-progress-card"
            role="status"
            aria-live="polite"
        >
            <div className="evaluation-progress-copy">
                <div className="evaluation-progress-heading">
                    <strong>{title}</strong>
                    <span>{percent}%</span>
                </div>
                <progress className="evaluation-progress-track" aria-label="Progreso de evaluación" max="100" value={percent} />
                <p>{message}</p>
            </div>
            {onCancel && (
                <button
                    type="button"
                    className="secondary-action evaluation-cancel-button"
                    onClick={onCancel}
                >
                    Cancelar
                </button>
            )}
        </section>
    )
}

export default EvaluationProgress
