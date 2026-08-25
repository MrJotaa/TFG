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

const SERIES_ORDER = ['Original', 'Tras el ataque']

function ComparisonTooltip({ active, label, payload }) {
    if (!active || !payload?.length) {
        return null
    }

    const valuesBySeries = Object.fromEntries(
        payload.map((entry) => [entry.dataKey, entry.value]),
    )

    return (
        <div className="evaluation-chart-tooltip">
            <strong>{label}</strong>
            {SERIES_ORDER.map((series) => (
                <div key={series} className={series.toLowerCase()}>
                    <span>{series}</span>
                    <strong>{Number(valuesBySeries[series] ?? 0).toFixed(1)} %</strong>
                </div>
            ))}
        </div>
    )
}

function ComparisonLegend() {
    return (
        <div className="evaluation-chart-legend" aria-label="Leyenda del gráfico">
            <span className="original">Original</span>
            <span className="adversarial">Tras el ataque</span>
        </div>
    )
}

function MetricComparisonChart({ aggregate }) {
    const data = [
        {
            metric: 'Exactitud',
            Original: aggregate.original.accuracy * 100,
            'Tras el ataque': aggregate.adversarial.accuracy * 100,
        },
        {
            metric: 'F1 macro',
            Original: aggregate.original.f1_macro * 100,
            'Tras el ataque': aggregate.adversarial.f1_macro * 100,
        },
        {
            metric: 'F1 ponderado',
            Original: aggregate.original.f1_weighted * 100,
            'Tras el ataque': aggregate.adversarial.f1_weighted * 100,
        },
        {
            metric: 'Recall macro',
            Original: aggregate.original.recall_macro * 100,
            'Tras el ataque': aggregate.adversarial.recall_macro * 100,
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
                        content={<ComparisonTooltip />}
                        cursor={{ fill: 'rgba(15, 23, 42, 0.04)' }}
                    />
                    <Legend content={<ComparisonLegend />} />
                    <Bar dataKey="Original" fill="var(--evaluation-original)" radius={[8, 8, 0, 0]} />
                    <Bar
                        dataKey="Tras el ataque"
                        fill="var(--evaluation-adversarial)"
                        radius={[8, 8, 0, 0]}
                    />
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}

export default MetricComparisonChart
