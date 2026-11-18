import { Fragment, useMemo, useState } from 'react'
import DetectionObjectImagePreview from './DetectionObjectImagePreview'
import { formatDecimal, formatInteger, formatRatio } from '../../utils/metricFormatting'
import {
    formatBoundingBox,
    getDetectionUnavailableMessage,
    getObjectOutcomeItems,
} from '../../utils/detectionMetrics'
import {
    DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
    DETECTION_OBJECT_PAGE_SIZES,
    filterAndSortDetectionObjects,
    getDetectionImageAggregateView,
    getDetectionObjectImagePreview,
    getDetectionObjectFilterOptions,
    paginateDetectionObjects,
} from '../../utils/detectionObjectExplorer'

const OUTCOME_LABELS = {
    preserved: 'Preservado',
    suppressed: 'Suprimido',
    recovered: 'Recuperado',
    missed_both: 'No detectado en ambas',
}

const SIZE_LABELS = {
    small: 'Pequeño (< 32²)',
    medium: 'Mediano (32²–96²)',
    large: 'Grande (≥ 96²)',
}

const SORT_OPTIONS = [
    { value: 'canonical', label: 'Imagen y objeto' },
    { value: 'confidence_drop', label: 'Mayor caída de confianza' },
    { value: 'iou_drop', label: 'Mayor caída de IoU' },
    { value: 'adversarial_confidence', label: 'Menor confianza tras el ataque' },
    { value: 'adversarial_iou', label: 'Menor IoU tras el ataque' },
    { value: 'class', label: 'Nombre de clase' },
]

function ObjectObservation({ observation }) {
    if (!observation) {
        return <span className="detection-object-not-detected">No detectado</span>
    }

    return (
        <span className="detection-object-observation">
            <strong>{formatDecimal(observation.confidence)}</strong>
            <small>conf. · IoU {formatDecimal(observation.iou_with_ground_truth)}</small>
        </span>
    )
}

function ObjectAggregateTable({ rows, dimension, onSelectImage }) {
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
                            <th scope="row">
                                {onSelectImage ? (
                                    <button
                                        type="button"
                                        className="detection-object-image-link"
                                        onClick={() => onSelectImage(row.image_id)}
                                        aria-label={`Inspeccionar imagen ${row.image_id}`}
                                        title="Ver imagen original y tras el ataque"
                                    >
                                        {row.label}
                                    </button>
                                ) : row.label}
                            </th>
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
    const [imageFilter, setImageFilter] = useState('all')
    const [categoryFilter, setCategoryFilter] = useState('all')
    const [sortBy, setSortBy] = useState('canonical')
    const [query, setQuery] = useState('')
    const [page, setPage] = useState(1)
    const [pageSize, setPageSize] = useState(DETECTION_OBJECT_PAGE_SIZES[0])
    const [expandedObjectId, setExpandedObjectId] = useState(null)
    const [previewSelection, setPreviewSelection] = useState(null)
    const [visibleImageRows, setVisibleImageRows] = useState(
        DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
    )
    const supervised = evaluation.supervised
    const analysis = supervised?.object_analysis
    const objects = useMemo(() => analysis?.objects ?? [], [analysis?.objects])

    const filterOptions = useMemo(
        () => getDetectionObjectFilterOptions(objects),
        [objects],
    )

    const filteredObjects = useMemo(() => {
        return filterAndSortDetectionObjects(objects, {
            query,
            outcome: outcomeFilter,
            imageId: imageFilter,
            categoryKey: categoryFilter,
            sortBy,
        })
    }, [categoryFilter, imageFilter, objects, outcomeFilter, query, sortBy])

    const pagination = useMemo(
        () => paginateDetectionObjects(filteredObjects, page, pageSize),
        [filteredObjects, page, pageSize],
    )

    function updateExplorerFilter(setter, value) {
        setter(value)
        setPage(1)
        setExpandedObjectId(null)
    }

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
    const imageAggregateView = getDetectionImageAggregateView(
        imageRows,
        visibleImageRows,
    )
    const imagePreview = previewSelection === null
        ? null
        : getDetectionObjectImagePreview(
            evaluation,
            previewSelection.imageId,
            previewSelection.objectId,
        )

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
                        <span className="evaluation-eyebrow">Emparejamiento por categoría</span>
                        <h3>Transiciones de cada objeto de referencia</h3>
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
                    <section className="evaluation-surface detection-table-surface detection-image-aggregate-section">
                        <div className="evaluation-section-heading">
                            <div>
                                <span className="evaluation-eyebrow">Detalle solicitado</span>
                                <h3>Resultados por imagen</h3>
                                <p>
                                    Se muestran en bloques de diez para mantener visible el resto del análisis por objeto.
                                </p>
                            </div>
                            <strong>
                                {formatInteger(imageAggregateView.visibleCount)} / {formatInteger(imageAggregateView.totalCount)}
                            </strong>
                        </div>
                        <ObjectAggregateTable
                            rows={imageAggregateView.items}
                            dimension="ID de imagen"
                            onSelectImage={(imageId) => setPreviewSelection({ imageId, objectId: null })}
                        />
                        {imageAggregateView.totalCount > DETECTION_IMAGE_AGGREGATE_BATCH_SIZE && (
                            <div className="detection-image-aggregate-actions">
                                {imageAggregateView.hasMore ? (
                                    <button
                                        type="button"
                                        className="secondary-action"
                                        onClick={() => setVisibleImageRows(imageAggregateView.nextVisibleCount)}
                                    >
                                        Ver 10 más
                                    </button>
                                ) : (
                                    <button
                                        type="button"
                                        className="text-action"
                                        onClick={() => setVisibleImageRows(DETECTION_IMAGE_AGGREGATE_BATCH_SIZE)}
                                    >
                                        Mostrar menos
                                    </button>
                                )}
                            </div>
                        )}
                    </section>

                    <section className="evaluation-surface detection-table-surface">
                        <div className="evaluation-section-heading">
                            <div>
                                <span className="evaluation-eyebrow">Universo materializado</span>
                                <h3>Explorador de objetos evaluados</h3>
                                <p>Filtra y ordena el universo completo; la tabla solo renderiza la página visible.</p>
                            </div>
                            <strong>{formatInteger(filteredObjects.length)} / {formatInteger(objects.length)}</strong>
                        </div>
                        <div className="detection-object-explorer-toolbar" aria-label="Filtros de objetos">
                            <label className="detection-object-search">
                                <span>Buscar</span>
                                <input
                                    type="search"
                                    value={query}
                                    placeholder="Objeto, imagen o clase"
                                    onChange={(event) => updateExplorerFilter(setQuery, event.target.value)}
                                />
                            </label>
                            <label>
                                <span>Transición</span>
                                <select value={outcomeFilter} onChange={(event) => updateExplorerFilter(setOutcomeFilter, event.target.value)}>
                                    <option value="all">Todas</option>
                                    {Object.entries(OUTCOME_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                                </select>
                            </label>
                            <label>
                                <span>Clase</span>
                                <select value={categoryFilter} onChange={(event) => updateExplorerFilter(setCategoryFilter, event.target.value)}>
                                    <option value="all">Todas las clases</option>
                                    {filterOptions.categories.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                                </select>
                            </label>
                            <label>
                                <span>Imagen</span>
                                <select value={imageFilter} onChange={(event) => updateExplorerFilter(setImageFilter, event.target.value)}>
                                    <option value="all">Todas las imágenes</option>
                                    {filterOptions.images.map((imageId) => <option key={imageId} value={imageId}>{imageId}</option>)}
                                </select>
                            </label>
                            <label>
                                <span>Ordenar por</span>
                                <select value={sortBy} onChange={(event) => updateExplorerFilter(setSortBy, event.target.value)}>
                                    {SORT_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                                </select>
                            </label>
                            <label>
                                <span>Objetos por página</span>
                                <select
                                    value={pageSize}
                                    onChange={(event) => updateExplorerFilter(setPageSize, Number(event.target.value))}
                                >
                                    {DETECTION_OBJECT_PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
                                </select>
                            </label>
                        </div>

                        {pagination.items.length ? (
                            <>
                                <div className="detection-object-range" role="status" aria-live="polite">
                                    Mostrando <strong>{formatInteger(pagination.start + 1)}–{formatInteger(pagination.end)}</strong> de <strong>{formatInteger(pagination.totalItems)}</strong> objetos filtrados
                                </div>
                                <div className="evaluation-table-wrapper">
                                    <table className="evaluation-table detection-objects-table">
                                        <thead>
                                            <tr>
                                                <th>Objeto</th>
                                                <th>Imagen</th>
                                                <th>Clase</th>
                                                <th>Transición</th>
                                                <th>Observación original</th>
                                                <th>Observación tras el ataque</th>
                                                <th>Detalle</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {pagination.items.map((object) => {
                                                const expanded = expandedObjectId === object.object_id
                                                return (
                                                    <Fragment key={object.object_id}>
                                                        <tr>
                                                            <td title={object.object_id}>
                                                                <button
                                                                    type="button"
                                                                    className="detection-object-image-link"
                                                                    aria-label={`Inspeccionar objeto ${object.object_id}`}
                                                                    title="Ver el objeto de referencia y sus detecciones emparejadas"
                                                                    onClick={() => setPreviewSelection({
                                                                        imageId: object.image_id,
                                                                        objectId: object.object_id,
                                                                    })}
                                                                >
                                                                    {object.object_id}
                                                                </button>
                                                            </td>
                                                            <td title={object.image_id}>
                                                                <button
                                                                    type="button"
                                                                    className="detection-object-image-link"
                                                                    aria-label={`Inspeccionar imagen ${object.image_id}`}
                                                                    title="Ver los objetos evaluados de esta imagen"
                                                                    onClick={() => setPreviewSelection({ imageId: object.image_id, objectId: null })}
                                                                >
                                                                    {object.image_id}
                                                                </button>
                                                            </td>
                                                            <td>{object.category.category_name}</td>
                                                            <td><span className={`detection-outcome-badge outcome-${object.outcome}`}>{OUTCOME_LABELS[object.outcome]}</span></td>
                                                            <td><ObjectObservation observation={object.original_observation} /></td>
                                                            <td><ObjectObservation observation={object.adversarial_observation} /></td>
                                                            <td>
                                                                <button
                                                                    type="button"
                                                                    className="detection-object-detail-button"
                                                                    aria-expanded={expanded}
                                                                    onClick={() => setExpandedObjectId(expanded ? null : object.object_id)}
                                                                >
                                                                    {expanded ? 'Ocultar' : 'Ver detalle'}
                                                                </button>
                                                            </td>
                                                        </tr>
                                                        {expanded && (
                                                            <tr className="detection-object-detail-row">
                                                                <td colSpan="7">
                                                                    <dl>
                                                                        <div><dt>Tamaño COCO</dt><dd>{SIZE_LABELS[object.size_bucket]}</dd></div>
                                                                        <div><dt>Área del objeto de referencia</dt><dd>{formatDecimal(object.area, 1)} px²</dd></div>
                                                                        <div><dt>Caja de referencia (xyxy)</dt><dd>{formatBoundingBox(object.ground_truth_box)}</dd></div>
                                                                        <div><dt>Caja original (xyxy)</dt><dd>{formatBoundingBox(object.original_observation?.detection_box)}</dd></div>
                                                                        <div><dt>Caja tras el ataque (xyxy)</dt><dd>{formatBoundingBox(object.adversarial_observation?.detection_box)}</dd></div>
                                                                        <div><dt>Δ confianza</dt><dd>{formatDecimal(object.confidence_drop)}</dd></div>
                                                                        <div><dt>Δ IoU</dt><dd>{formatDecimal(object.iou_drop)}</dd></div>
                                                                    </dl>
                                                                </td>
                                                            </tr>
                                                        )}
                                                    </Fragment>
                                                )
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                                {pagination.totalPages > 1 && (
                                    <nav className="evaluation-pagination" aria-label="Paginación de objetos evaluados">
                                        <button type="button" disabled={pagination.page === 1} onClick={() => { setPage(1); setExpandedObjectId(null) }}>Primera</button>
                                        <button type="button" disabled={pagination.page === 1} onClick={() => { setPage((current) => Math.max(1, current - 1)); setExpandedObjectId(null) }}>Anterior</button>
                                        <span>Página <strong aria-current="page">{pagination.page}</strong> de <strong>{pagination.totalPages}</strong></span>
                                        <button type="button" disabled={pagination.page === pagination.totalPages} onClick={() => { setPage((current) => Math.min(pagination.totalPages, current + 1)); setExpandedObjectId(null) }}>Siguiente</button>
                                        <button type="button" disabled={pagination.page === pagination.totalPages} onClick={() => { setPage(pagination.totalPages); setExpandedObjectId(null) }}>Última</button>
                                    </nav>
                                )}
                            </>
                        ) : (
                            <div className="evaluation-empty-panel compact">
                                <h3>No hay objetos para estos filtros</h3>
                                <p>Modifica la búsqueda, la clase, la imagen o la transición.</p>
                            </div>
                        )}
                    </section>
                </>
            ) : (
                <div className="evaluation-warning-banner detection-details-banner">
                    <strong>Respuesta compacta.</strong>
                    <span>Los agregados globales, por clase y por tamaño están completos. Activa “Incluir detalle por objeto” antes de una nueva ejecución para materializar las tablas por imagen y objeto.</span>
                </div>
            )}
            {imagePreview && (
                <DetectionObjectImagePreview
                    key={`${imagePreview.imageId}:${imagePreview.selectedObject?.object_id ?? 'image'}`}
                    image={imagePreview}
                    onShowImage={() => setPreviewSelection({ imageId: imagePreview.imageId, objectId: null })}
                    onClose={() => setPreviewSelection(null)}
                />
            )}
        </div>
    )
}

export default DetectionObjectAnalysis
