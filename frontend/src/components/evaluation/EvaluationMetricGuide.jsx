function EvaluationMetricGuide({ guide, variant = null }) {
    if (!guide) {
        return null
    }

    const variantClassName = variant ? ` ${variant}-guide` : ''

    return (
        <details className={`evaluation-metric-guide${variantClassName}`}>
            <summary>
                <span className="evaluation-info-icon" aria-hidden="true">i</span>
                <span>
                    <strong>Guía de esta pestaña</strong>
                    <small>{guide.title}</small>
                </span>
                <span className="evaluation-guide-chevron" aria-hidden="true">⌄</span>
            </summary>
            <div className="evaluation-metric-guide-body">
                <div className="evaluation-metric-guide-body-inner">
                    <p className="evaluation-metric-guide-intro">{guide.summary}</p>
                    <dl className="evaluation-metric-guide-grid">
                        {guide.items.map((item) => (
                            <div key={item.term}>
                                <dt>{item.term}</dt>
                                <dd>{item.definition}</dd>
                            </div>
                        ))}
                    </dl>
                    {guide.note && (
                        <p className="evaluation-metric-guide-note">
                            <strong>Clave de interpretación:</strong> {guide.note}
                        </p>
                    )}
                </div>
            </div>
        </details>
    )
}

export default EvaluationMetricGuide
