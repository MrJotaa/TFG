import {
    evaluationProgressMessage,
    evaluationProgressPercent,
} from '../../utils/evaluationProgress'

function EvaluationProgress({ progress, title, pendingMessage }) {
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
                <div
                    className="evaluation-progress-track"
                    role="progressbar"
                    aria-label="Progreso de evaluación"
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-valuenow={percent}
                >
                    <div
                        className="evaluation-progress-value"
                        style={{ width: `${percent}%` }}
                    />
                </div>
                <p>{message}</p>
            </div>
        </section>
    )
}

export default EvaluationProgress
