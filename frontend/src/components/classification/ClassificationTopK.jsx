function formatProbability(value, digits = 2) {
    const numericValue = Number(value)

    if (!Number.isFinite(numericValue)) {
        return '—'
    }

    return `${(numericValue * 100).toFixed(digits)}%`
}

function ClassificationTopK({ title, predictions = [] }) {
    return (
        <section className="classification-topk-column">
            <h4>{title}</h4>
            <div className="classification-topk-list">
                {predictions.map((prediction) => {
                    const probability = Number(prediction.probability)
                    const width = Number.isFinite(probability)
                        ? `${Math.max(0, Math.min(100, probability * 100))}%`
                        : '0%'

                    return (
                        <div key={prediction.label} className="classification-topk-row">
                            <div>
                                <span>{prediction.label}</span>
                                <strong>{formatProbability(prediction.probability)}</strong>
                            </div>
                            <span className="classification-topk-track" aria-hidden="true">
                                <span style={{ width }} />
                            </span>
                        </div>
                    )
                })}
            </div>
        </section>
    )
}

export default ClassificationTopK
