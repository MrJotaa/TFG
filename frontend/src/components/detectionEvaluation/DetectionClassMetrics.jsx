import { useMemo, useState } from 'react'
import { formatInteger, formatRatio } from '../../utils/metricFormatting'
import {
    findClassMetrics,
    formatSignedPercentagePoints,
    getDetectionUnavailableMessage,
} from '../../utils/detectionMetrics'

function DetectionClassMetrics({ evaluation }) {
    const [query, setQuery] = useState('')
    const [sort, setSort] = useState('support')
    const supervised = evaluation.supervised

    const rows = useMemo(() => {
        if (!supervised?.available) {
            return []
        }

        const normalizedQuery = query.trim().toLocaleLowerCase('es')
        const objectClasses = supervised.object_analysis?.by_class ?? []

        return supervised.original.per_class
            .map((original) => {
                const classId = original.reference.model_class_id
                return {
                    reference: original.reference,
                    original,
                    adversarial: findClassMetrics(supervised.adversarial, classId),
                    delta: supervised.deltas.per_class.find(
                        (entry) => entry.reference.model_class_id === classId,
                    ),
                    objects: objectClasses.find(
                        (entry) => entry.category.model_class_id === classId,
                    ),
                }
            })
            .filter((row) => {
                if (!normalizedQuery) {
                    return true
                }
                return [
                    row.reference.category_name,
                    row.reference.model_class_name,
                    String(row.reference.model_class_id),
                ].some((value) => value.toLocaleLowerCase('es').includes(normalizedQuery))
            })
            .sort((left, right) => {
                if (sort === 'name') {
                    return left.reference.category_name.localeCompare(
                        right.reference.category_name,
                        'es',
                    )
                }
                if (sort === 'degradation') {
                    return (
                        (right.delta?.average_precision.absolute ?? -Infinity) -
                        (left.delta?.average_precision.absolute ?? -Infinity)
                    )
                }
                return right.original.support - left.original.support
            })
    }, [query, sort, supervised])

    if (!supervised?.available) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Análisis por clase</span>
                <h3>No hay métricas por clase disponibles</h3>
                <p>{getDetectionUnavailableMessage(supervised?.reason)}</p>
            </div>
        )
    }

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-surface class-metrics-header">
                <div>
                    <span className="evaluation-eyebrow">Desglose por categoría</span>
                    <h3>Rendimiento y transiciones por clase</h3>
                    <p>
                        Compara AP, AR y el punto de operación con el número de objetos de referencia de cada categoría resuelta.
                    </p>
                </div>
                <div className="class-metrics-summary">
                    <div><strong>{formatInteger(supervised.original.per_class.length)}</strong><span>Clases</span></div>
                    <div><strong>{formatInteger(supervised.annotation_coverage.evaluable_objects)}</strong><span>Objetos</span></div>
                    <div className="warning"><strong>{formatInteger(supervised.object_analysis.summary.suppressed_count)}</strong><span>Suprimidos</span></div>
                </div>
            </section>

            <section className="class-metrics-toolbar detection-class-toolbar">
                <label>
                    <span>Buscar clase</span>
                    <input
                        type="search"
                        value={query}
                        placeholder="Nombre o ID del modelo"
                        onChange={(event) => setQuery(event.target.value)}
                    />
                </label>
                <label>
                    <span>Ordenar por</span>
                    <select value={sort} onChange={(event) => setSort(event.target.value)}>
                        <option value="support">Mayor soporte</option>
                        <option value="degradation">Mayor degradación AP</option>
                        <option value="name">Nombre</option>
                    </select>
                </label>
            </section>

            <section className="evaluation-surface detection-table-surface">
                <div className="evaluation-table-wrapper">
                    <table className="evaluation-table detection-class-table">
                        <thead>
                            <tr>
                                <th>Clase</th>
                                <th>Soporte</th>
                                <th>AP orig.</th>
                                <th>AP tras ataque</th>
                                <th>Δ AP</th>
                                <th>AR orig.</th>
                                <th>AR tras ataque</th>
                                <th>TP / FP / FN orig.</th>
                                <th>TP / FP / FN tras ataque</th>
                                <th>Preservados</th>
                                <th>Suprimidos</th>
                                <th>Recuperados</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row) => (
                                <tr key={row.reference.category_internal_id}>
                                    <th scope="row">
                                        <strong>{row.reference.category_name}</strong>
                                        <small>
                                            Modelo: {row.reference.model_class_name} · ID {row.reference.model_class_id}
                                        </small>
                                    </th>
                                    <td>{formatInteger(row.original.support)}</td>
                                    <td>{formatRatio(row.original.average_precision)}</td>
                                    <td>{formatRatio(row.adversarial?.average_precision)}</td>
                                    <td>{formatSignedPercentagePoints(row.delta?.average_precision.absolute)}</td>
                                    <td>{formatRatio(row.original.average_recall)}</td>
                                    <td>{formatRatio(row.adversarial?.average_recall)}</td>
                                    <td>{row.original.operating_point.true_positives} / {row.original.operating_point.false_positives} / {row.original.operating_point.false_negatives}</td>
                                    <td>{row.adversarial?.operating_point.true_positives} / {row.adversarial?.operating_point.false_positives} / {row.adversarial?.operating_point.false_negatives}</td>
                                    <td>{formatInteger(row.objects?.preserved_count)}</td>
                                    <td className="tone-danger">{formatInteger(row.objects?.suppressed_count)}</td>
                                    <td className="tone-success">{formatInteger(row.objects?.recovered_count)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                {!rows.length && <p className="detection-no-results">No hay clases que coincidan con la búsqueda.</p>}
            </section>
        </div>
    )
}

export default DetectionClassMetrics
