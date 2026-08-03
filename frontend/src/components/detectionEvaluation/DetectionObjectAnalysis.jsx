import { useMemo, useState } from 'react'
import { formatDecimal, formatInteger, formatRatio } from '../../utils/metricFormatting'
import {
    formatBoundingBox,
    getDetectionUnavailableMessage,
    getObjectOutcomeItems,
} from '../../utils/detectionMetrics'

const OUTCOME_LABELS = {
    preserved: 'Preservado',
    suppressed: 'Suprimido',
    recovered: 'Recuperado',
    missed_both: 'No detectado en ambas',
}

const SIZE_LABELS = {
    small: 'Small (< 32²)',
    medium: 'Medium (32²–96²)',
    large: 'Large (≥ 96²)',
}

function ObjectAggregateTable({ rows, dimension }) {
    return (
        <div className="evaluation-table-wrapper">
            <table className="evaluation-table detection-object-aggregate-table">
                <thead>
                    <tr>
                        <th>{dimension}</th>
                        <th>Total</th>
                        <th>Preservados</th>
                        <th>Suprimidos</th>
                        <th>Recuperados</th>
                        <th>No detectados</th>
                        <th>Preservación</th>
                        <th>Supresión</th>
                        <th>Δ confianza</th>
                        <th>Δ IoU</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => (
                        <tr key={row.key}>
                            <th scope="row">{row.label}</th>
                            <td>{formatInteger(row.total_objects)}</td>
                            <td>{formatInteger(row.preserved_count)}</td>
                            <td className="tone-danger">{formatInteger(row.suppressed_count)}</td>
                            <td className="tone-success">{formatInteger(row.recovered_count)}</td>
                            <td>{formatInteger(row.missed_both_count)}</td>
                            <td>{formatRatio(row.preservation_rate)}</td>
                            <td>{formatRatio(row.suppression_rate)}</td>
                            <td>{formatDecimal(row.mean_confidence_drop_preserved)}</td>
                            <td>{formatDecimal(row.mean_iou_drop_preserved)}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

function DetectionObjectAnalysis({ evaluation }) {
    const [outcomeFilter, setOutcomeFilter] = useState('all')
    const [query, setQuery] = useState('')
    const supervised = evaluation.supervised
    const analysis = supervised?.object_analysis

    const filteredObjects = useMemo(() => {
        const normalizedQuery = query.trim().toLocaleLowerCase('es')
        return (analysis?.objects ?? []).filter((object) => {
            if (outcomeFilter !== 'all' && object.outcome !== outcomeFilter) {
                return false
            }
            if (!normalizedQuery) {
                return true
            }
            return [object.object_id, object.image_id, object.category.category_name].some(
                (value) => String(value).toLocaleLowerCase('es').includes(normalizedQuery),
            )
        })
    }, [analysis?.objects, outcomeFilter, query])

    if (!supervised?.available || !analysis) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Análisis por objeto</span>
                <h3>Las transiciones no están disponibles</h3>
                <p>{getDetectionUnavailableMessage(supervised?.reason)}</p>
            </div>
        )
    }

    const outcomes = getObjectOutcomeItems(analysis.summary)
    const sizeRows = analysis.by_size.map((row) => ({
        ...row,
        key: row.size_bucket,
        label: SIZE_LABELS[row.size_bucket],
    }))
    const imageRows = (analysis.by_image ?? []).map((row) => ({
        ...row,
        key: row.image_id,
        label: row.image_id,
    }))

    return (
        <div className="evaluation-tab-content">
            <section className="evaluation-kpi-grid">
                {outcomes.map((item) => (
                    <article key={item.field} className={`evaluation-metric-card detection-outcome-metric tone-${item.tone}`}>
                        <div className="evaluation-metric-card-heading"><span>{item.label}</span></div>
                        <strong>{formatInteger(item.count)}</strong>
                        <p>{formatRatio(analysis.summary.total_objects ? item.count / analysis.summary.total_objects : null)} del universo evaluable</p>
                    </article>
                ))}
            </section>

            <section className="evaluation-surface detection-method-card">
                <div className="evaluation-section-heading">
                    <div>
                        <span className="evaluation-eyebrow">Método category-aware</span>
                        <h3>Transiciones de cada ground truth</h3>
                    </div>
                    <span className="detection-methodology-badge">{analysis.methodology_version}</span>
                </div>
                <p>
                    Cada objeto evaluable se clasifica una sola vez usando los matches oficiales de COCOeval en el punto de operación: confianza ≥ {analysis.operating_point_configuration.confidence_threshold} e IoU ≥ {analysis.operating_point_configuration.iou_threshold}.
                </p>
            </section>

            <section className="evaluation-surface detection-table-surface">
                <div className="evaluation-section-heading"><div><span className="evaluation-eyebrow">Escala COCO</span><h3>Resultados por tamaño</h3></div></div>
                <ObjectAggregateTable rows={sizeRows} dimension="Tamaño" />
            </section>

            {analysis.details_included ? (
                <>
                    <section className="evaluation-surface detection-table-surface">
                        <div className="evaluation-section-heading"><div><span className="evaluation-eyebrow">Detalle solicitado</span><h3>Resultados por imagen</h3></div></div>
                        <ObjectAggregateTable rows={imageRows} dimension="ID de imagen" />
                    </section>

                    <section className="evaluation-surface detection-table-surface">
                        <div className="evaluation-section-heading">
                            <div><span className="evaluation-eyebrow">Universo materializado</span><h3>Objetos evaluados</h3></div>
                            <strong>{formatInteger(filteredObjects.length)} / {formatInteger(analysis.objects.length)}</strong>
                        </div>
                        <div className="detection-object-toolbar">
                            <label><span>Buscar</span><input type="search" value={query} placeholder="Objeto, imagen o clase" onChange={(event) => setQuery(event.target.value)} /></label>
                            <label><span>Outcome</span><select value={outcomeFilter} onChange={(event) => setOutcomeFilter(event.target.value)}><option value="all">Todos</option>{Object.entries(OUTCOME_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
                        </div>
                        <div className="evaluation-table-wrapper">
                            <table className="evaluation-table detection-objects-table">
                                <thead><tr><th>Objeto</th><th>Imagen</th><th>Clase</th><th>Tamaño</th><th>Outcome</th><th>GT xyxy</th><th>Conf. orig.</th><th>Conf. adv.</th><th>IoU orig.</th><th>IoU adv.</th><th>Δ conf.</th><th>Δ IoU</th></tr></thead>
                                <tbody>
                                    {filteredObjects.map((object) => (
                                        <tr key={object.object_id}>
                                            <td title={object.object_id}>{object.object_id}</td>
                                            <td title={object.image_id}>{object.image_id}</td>
                                            <td>{object.category.category_name}</td>
                                            <td>{SIZE_LABELS[object.size_bucket]}</td>
                                            <td><span className={`detection-outcome-badge outcome-${object.outcome}`}>{OUTCOME_LABELS[object.outcome]}</span></td>
                                            <td>{formatBoundingBox(object.ground_truth_box)}</td>
                                            <td>{formatDecimal(object.original_observation?.confidence)}</td>
                                            <td>{formatDecimal(object.adversarial_observation?.confidence)}</td>
                                            <td>{formatDecimal(object.original_observation?.iou_with_ground_truth)}</td>
                                            <td>{formatDecimal(object.adversarial_observation?.iou_with_ground_truth)}</td>
                                            <td>{formatDecimal(object.confidence_drop)}</td>
                                            <td>{formatDecimal(object.iou_drop)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </section>
                </>
            ) : (
                <div className="evaluation-warning-banner detection-details-banner">
                    <strong>Respuesta compacta.</strong>
                    <span>Los agregados globales, por clase y por tamaño están completos. Activa “Incluir detalle por objeto” antes de una nueva ejecución para materializar las tablas por imagen y objeto.</span>
                </div>
            )}
        </div>
    )
}

export default DetectionObjectAnalysis
