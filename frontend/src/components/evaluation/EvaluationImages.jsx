import { useId, useMemo, useState } from 'react'
import {
    formatDecimal,
    formatInteger,
    formatRatio,
} from '../../utils/metricFormatting'
import { resolveApiUrl } from '../../api/client'
import {
    getEvaluationGradCamEntry,
    getEvaluationGradCamKey,
} from '../../utils/evaluationGradCamState'

const PAGE_SIZE = 6

const FILTERS = [
    { id: 'all', label: 'Todas' },
    { id: 'correct_to_incorrect', label: 'Correcta → incorrecta' },
    { id: 'incorrect_to_correct', label: 'Incorrecta → correcta' },
    { id: 'changed', label: 'Predicción modificada' },
    { id: 'unchanged', label: 'Sin cambio' },
    { id: 'unresolved', label: 'Sin etiqueta válida' },
]

const TRANSITION_LABELS = {
    correct_to_correct: {
        label: 'Se mantiene correcta',
        tone: 'success',
    },
    correct_to_incorrect: {
        label: 'Correcta → incorrecta',
        tone: 'danger',
    },
    incorrect_to_correct: {
        label: 'Incorrecta → correcta',
        tone: 'success',
    },
    incorrect_to_incorrect: {
        label: 'Sigue incorrecta',
        tone: 'warning',
    },
}

function getStatus(result) {
    const transition = result.diagnostics?.transition

    if (transition && TRANSITION_LABELS[transition]) {
        return TRANSITION_LABELS[transition]
    }

    if (result.diagnostics?.label_resolution_reason) {
        return {
            label: 'Sin etiqueta válida',
            tone: 'warning',
        }
    }

    return result.prediction_changed
        ? { label: 'Predicción modificada', tone: 'danger' }
        : { label: 'Sin cambio', tone: 'neutral' }
}

function matchesFilter(result, filter) {
    if (filter === 'all') {
        return true
    }

    if (filter === 'changed') {
        return result.prediction_changed
    }

    if (filter === 'unchanged') {
        return !result.prediction_changed
    }

    if (filter === 'unresolved') {
        return Boolean(result.diagnostics?.label_resolution_reason)
    }

    return result.diagnostics?.transition === filter
}

function formatSignedRatio(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '—'
    }

    const prefix = value > 0 ? '+' : ''
    return `${prefix}${formatRatio(value)}`
}

function BooleanValue({ value }) {
    if (value === null || value === undefined) {
        return '—'
    }

    return value ? 'Sí' : 'No'
}

function ImagePreview({ label, path, alt, variant }) {
    const url = resolveApiUrl(path)

    return (
        <figure className={`evaluation-image-preview ${variant}`}>
            <figcaption>{label}</figcaption>
            {url ? (
                <img src={url} alt={alt} loading="lazy" />
            ) : (
                <div className="evaluation-image-missing">Imagen no disponible</div>
            )}
        </figure>
    )
}

function EvaluationImageCard({
    evaluationId,
    result,
    gradCamEntry,
    onGenerateGradCam,
}) {
    const [expanded, setExpanded] = useState(false)
    const titleId = useId()
    const diagnosticsId = useId()
    const gradCamErrorId = useId()
    const diagnostics = result.diagnostics ?? {}
    const status = getStatus(result)
    const gradCamLoading = gradCamEntry.status === 'loading'
    const gradCamReady = gradCamEntry.status === 'success'
    const gradCamError = gradCamEntry.status === 'error'
        ? gradCamEntry.error
        : ''

    return (
        <article className="evaluation-image-card" aria-labelledby={titleId}>
            <header className="evaluation-image-card-header">
                <div>
                    <span className="evaluation-eyebrow">{result.filename}</span>
                    <h4 id={titleId}>
                        {result.original_label} <span aria-hidden="true">→</span>{' '}
                        {result.adversarial_label}
                    </h4>
                </div>

                <span
                    className={`evaluation-status-badge ${status.tone}`}
                    aria-label={`Estado: ${status.label}`}
                >
                    {status.label}
                </span>
            </header>

            <div className="evaluation-image-card-body">
                <div className="evaluation-image-triptych">
                    <ImagePreview
                        label="Original"
                        path={result.original_image_url}
                        alt={`Imagen original ${result.filename}`}
                        variant="original"
                    />
                    <ImagePreview
                        label="Tras el ataque"
                        path={result.adversarial_image_url}
                        alt={`Imagen tras el ataque ${result.filename}`}
                        variant="adversarial"
                    />
                    <ImagePreview
                        label="Perturbación"
                        path={result.perturbation_image_url}
                        alt={`Perturbación aplicada a ${result.filename}`}
                        variant="perturbation"
                    />
                </div>

                <div className="evaluation-image-summary">
                    <dl>
                        <div>
                            <dt>Clase de referencia</dt>
                            <dd>{result.ground_truth_label ?? 'Sin etiqueta'}</dd>
                        </div>
                        <div>
                            <dt>Confianza original</dt>
                            <dd>{formatRatio(result.original_confidence)}</dd>
                        </div>
                        <div>
                            <dt>Confianza tras el ataque</dt>
                            <dd>{formatRatio(result.adversarial_confidence)}</dd>
                        </div>
                        <div>
                            <dt>Clase real: caída</dt>
                            <dd>
                                {formatSignedRatio(
                                    diagnostics.true_class_probability_drop,
                                )}
                            </dd>
                        </div>
                    </dl>

                    <button
                        type="button"
                        className="evaluation-detail-button"
                        aria-expanded={expanded}
                        aria-controls={diagnosticsId}
                        onClick={() => setExpanded((current) => !current)}
                    >
                        {expanded ? 'Ocultar diagnóstico' : 'Ver diagnóstico completo'}
                    </button>
                </div>
            </div>

            <section
                className="evaluation-gradcam-section"
                aria-label={`Grad-CAM de ${result.filename}`}
            >
                <div className="evaluation-gradcam-action">
                    <div>
                        <strong>Explicabilidad Grad-CAM</strong>
                        <span>
                            Genera los mapas únicamente para esta imagen evaluada.
                        </span>
                    </div>
                    <button
                        type="button"
                        className="evaluation-gradcam-button"
                        disabled={gradCamLoading || gradCamReady}
                        aria-describedby={gradCamError ? gradCamErrorId : undefined}
                        onClick={() =>
                            onGenerateGradCam(evaluationId, result.image_id)
                        }
                    >
                        {gradCamLoading
                            ? 'Generando Grad-CAM…'
                            : gradCamReady
                              ? 'Grad-CAM generado'
                              : gradCamError
                                ? 'Reintentar Grad-CAM'
                                : 'Generar Grad-CAM'}
                    </button>
                </div>

                {gradCamLoading && (
                    <p
                        className="evaluation-gradcam-status"
                        role="status"
                        aria-live="polite"
                    >
                        Generando Grad-CAM original y tras el ataque para esta imagen.
                    </p>
                )}

                {gradCamError && (
                    <div
                        id={gradCamErrorId}
                        className="evaluation-gradcam-error"
                        role="alert"
                    >
                        {gradCamError}
                    </div>
                )}

                {gradCamReady && (
                    <div className="evaluation-gradcam-grid">
                        <ImagePreview
                            label="Grad-CAM original"
                            path={
                                gradCamEntry.data
                                    .original_gradcam_image_url
                            }
                            alt={`Grad-CAM original de ${result.filename}`}
                            variant="gradcam-original"
                        />
                        <ImagePreview
                            label="Grad-CAM tras el ataque"
                            path={
                                gradCamEntry.data
                                    .adversarial_gradcam_image_url
                            }
                            alt={`Grad-CAM tras el ataque de ${result.filename}`}
                            variant="gradcam-adversarial"
                        />
                    </div>
                )}
            </section>

            <div
                id={diagnosticsId}
                className="evaluation-image-diagnostics"
                hidden={!expanded}
            >
                    <div>
                        <span>Prob. clase real original</span>
                        <strong>
                            {formatRatio(diagnostics.original_true_class_probability)}
                        </strong>
                    </div>
                    <div>
                        <span>Prob. clase real tras el ataque</span>
                        <strong>
                            {formatRatio(diagnostics.adversarial_true_class_probability)}
                        </strong>
                    </div>
                    <div>
                        <span>Margen original</span>
                        <strong>{formatDecimal(diagnostics.original_prediction_margin)}</strong>
                    </div>
                    <div>
                        <span>Margen tras el ataque</span>
                        <strong>
                            {formatDecimal(diagnostics.adversarial_prediction_margin)}
                        </strong>
                    </div>
                    <div>
                        <span>Caída del margen</span>
                        <strong>{formatDecimal(diagnostics.prediction_margin_drop)}</strong>
                    </div>
                    <div>
                        <span>Top-5 original</span>
                        <strong>
                            <BooleanValue value={diagnostics.original_top_5_correct} />
                        </strong>
                    </div>
                    <div>
                        <span>Top-5 tras el ataque</span>
                        <strong>
                            <BooleanValue value={diagnostics.adversarial_top_5_correct} />
                        </strong>
                    </div>
                    <div>
                        <span>Perturbación L∞</span>
                        <strong>{formatDecimal(result.perturbation_linf)}</strong>
                    </div>
                    <div>
                        <span>Perturbación L2</span>
                        <strong>{formatDecimal(result.perturbation_l2)}</strong>
                    </div>
                </div>

            {diagnostics.label_resolution_reason && (
                <div className="evaluation-image-warning">
                    <strong>Etiqueta no resuelta:</strong>{' '}
                    {diagnostics.label_resolution_reason}
                </div>
            )}
        </article>
    )
}

function EvaluationImages({
    evaluation,
    gradCamStates,
    onGenerateGradCam,
}) {
    const [filter, setFilter] = useState('all')
    const [search, setSearch] = useState('')
    const [page, setPage] = useState(1)

    const results = useMemo(() => evaluation?.results ?? [], [evaluation])

    const filteredResults = useMemo(() => {
        const normalizedSearch = search.trim().toLocaleLowerCase('es')

        return results.filter((result) => {
            if (!matchesFilter(result, filter)) {
                return false
            }

            if (!normalizedSearch) {
                return true
            }

            const searchableText = [
                result.filename,
                result.ground_truth_label,
                result.original_label,
                result.adversarial_label,
            ]
                .filter(Boolean)
                .join(' ')
                .toLocaleLowerCase('es')

            return searchableText.includes(normalizedSearch)
        })
    }, [filter, results, search])

    const totalPages = Math.max(1, Math.ceil(filteredResults.length / PAGE_SIZE))
    const safePage = Math.min(page, totalPages)
    const pageStart = (safePage - 1) * PAGE_SIZE
    const pageResults = filteredResults.slice(pageStart, pageStart + PAGE_SIZE)

    const changedCount = results.filter((result) => result.prediction_changed).length
    const degradedCount = results.filter(
        (result) => result.diagnostics?.transition === 'correct_to_incorrect',
    ).length

    if (!results.length) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Resultados individuales</span>
                <h3>No hay imágenes evaluadas</h3>
                <p>La respuesta no contiene resultados detallados por imagen.</p>
            </div>
        )
    }

    return (
        <div className="evaluation-tab-content evaluation-images-tab">
            <section className="evaluation-surface evaluation-images-header">
                <div>
                    <span className="evaluation-eyebrow">Exploración visual</span>
                    <h3>Resultados por imagen</h3>
                    <p>
                        Compara original, resultado tras el ataque y perturbación, y filtra los casos que
                        explican la degradación global.
                    </p>
                </div>

                <div className="evaluation-images-summary" aria-label="Resumen de imágenes">
                    <div>
                        <strong>{formatInteger(results.length)}</strong>
                        <span>Evaluadas</span>
                    </div>
                    <div>
                        <strong>{formatInteger(changedCount)}</strong>
                        <span>Modificadas</span>
                    </div>
                    <div className={degradedCount ? 'danger' : ''}>
                        <strong>{formatInteger(degradedCount)}</strong>
                        <span>Degradadas</span>
                    </div>
                </div>
            </section>

            <section className="evaluation-images-toolbar" aria-label="Filtros de imágenes">
                <div className="evaluation-image-filter-list">
                    {FILTERS.map((option) => (
                        <button
                            key={option.id}
                            type="button"
                            className={filter === option.id ? 'active' : ''}
                            aria-pressed={filter === option.id}
                            onClick={() => {
                                setFilter(option.id)
                                setPage(1)
                            }}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>

                <label>
                    <span>Buscar</span>
                    <input
                        type="search"
                        value={search}
                        onChange={(event) => {
                            setSearch(event.target.value)
                            setPage(1)
                        }}
                        placeholder="Archivo, clase o etiqueta"
                    />
                </label>
            </section>

            <div className="evaluation-images-count" role="status" aria-live="polite">
                <strong>{formatInteger(filteredResults.length)}</strong> resultados encontrados
            </div>

            {pageResults.length ? (
                <div className="evaluation-image-list" aria-label="Resultados por imagen">
                    {pageResults.map((result) => (
                        <EvaluationImageCard
                            key={result.image_id}
                            evaluationId={evaluation.result_id}
                            result={result}
                            gradCamEntry={getEvaluationGradCamEntry(
                                gradCamStates,
                                getEvaluationGradCamKey(
                                    evaluation.result_id,
                                    result.image_id,
                                ),
                            )}
                            onGenerateGradCam={onGenerateGradCam}
                        />
                    ))}
                </div>
            ) : (
                <div className="evaluation-empty-panel compact">
                    <h3>No hay resultados para este filtro</h3>
                    <p>Prueba con otra transición o modifica el texto de búsqueda.</p>
                </div>
            )}

            {totalPages > 1 && (
                <nav className="evaluation-pagination" aria-label="Paginación de imágenes">
                    <button
                        type="button"
                        disabled={safePage === 1}
                        onClick={() => setPage((current) => Math.max(1, current - 1))}
                    >
                        Anterior
                    </button>
                    <span aria-live="polite">
                        Página <strong aria-current="page">{safePage}</strong> de{' '}
                        <strong>{totalPages}</strong>
                    </span>
                    <button
                        type="button"
                        disabled={safePage === totalPages}
                        onClick={() =>
                            setPage((current) => Math.min(totalPages, current + 1))
                        }
                    >
                        Siguiente
                    </button>
                </nav>
            )}
        </div>
    )
}

export default EvaluationImages
