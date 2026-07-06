import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts'

function MetricComparisonChart({ aggregate }) {
    const data = [
        {
            metric: 'Accuracy',
            Original: aggregate.original.accuracy * 100,
            Adversarial: aggregate.adversarial.accuracy * 100,
        },
        {
            metric: 'F1 macro',
            Original: aggregate.original.f1_macro * 100,
            Adversarial: aggregate.adversarial.f1_macro * 100,
        },
        {
            metric: 'F1 weighted',
            Original: aggregate.original.f1_weighted * 100,
            Adversarial: aggregate.adversarial.f1_weighted * 100,
        },
        {
            metric: 'Recall macro',
            Original: aggregate.original.recall_macro * 100,
            Adversarial: aggregate.adversarial.recall_macro * 100,
        },
    ]

    return (
        <div className="evaluation-chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" vertical={false} />
                    <XAxis dataKey="metric" tickLine={false} axisLine={false} />
                    <YAxis
                        domain={[0, 100]}
                        tickFormatter={(value) => `${value}%`}
                        tickLine={false}
                        axisLine={false}
                    />
                    <Tooltip
                        formatter={(value) => [`${Number(value).toFixed(1)} %`]}
                        cursor={{ fill: 'rgba(15, 23, 42, 0.04)' }}
                    />
                    <Legend iconType="circle" />
                    <Bar dataKey="Original" fill="var(--evaluation-original)" radius={[8, 8, 0, 0]} />
                    <Bar
                        dataKey="Adversarial"
                        fill="var(--evaluation-adversarial)"
                        radius={[8, 8, 0, 0]}
                    />
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}

export default MetricComparisonChart
