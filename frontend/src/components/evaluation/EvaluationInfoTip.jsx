import { useId } from 'react'

function EvaluationInfoTip({ label, children }) {
    const tooltipId = useId()

    if (!children) return null

    return (
        <span className="evaluation-info-tip">
            <button
                type="button"
                className="evaluation-info-tip-trigger"
                aria-label={`Más información sobre ${label}`}
                aria-describedby={tooltipId}
            >
                i
            </button>
            <span id={tooltipId} className="evaluation-info-tip-content" role="tooltip">
                <strong>{label}</strong>
                <span>{children}</span>
            </span>
        </span>
    )
}

export default EvaluationInfoTip
