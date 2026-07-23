function MetricCard({ label, value, helper, tone = 'neutral', badge }) {
    return (
        <article className={`evaluation-metric-card tone-${tone}`}>
            <div className="evaluation-metric-card-heading">
                <span>{label}</span>
                {badge && <small>{badge}</small>}
            </div>

            <strong>{value}</strong>

            {helper && <p>{helper}</p>}
        </article>
    )
}

export default MetricCard
