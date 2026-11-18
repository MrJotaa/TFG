import EvaluationInfoTip from './EvaluationInfoTip'

function MetricCard({ label, value, helper, tone = 'neutral', badge, info }) {
    return (
        <article className={`evaluation-metric-card tone-${tone}`}>
            <div className="evaluation-metric-card-heading">
                <span>
                    {label}
                    {info && <EvaluationInfoTip label={label}>{info}</EvaluationInfoTip>}
                </span>
                {badge && <small>{badge}</small>}
            </div>

            <strong>{value}</strong>

            {helper && <p>{helper}</p>}
        </article>
    )
}

export default MetricCard
