import { useEffect, useState } from 'react'
import { resolveApiUrl } from '../api/client'
import {
    deleteEvaluation,
    getAllEvaluationHistory,
    getEvaluationHistory,
    getStoredEvaluation,
    setEvaluationPreserved,
} from '../api/evaluationHistory'
import {
    deleteIndividualAnalysis,
    getIndividualAnalysisHistory,
    getStoredIndividualAnalysis,
} from '../api/individualAnalysisHistory'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import { getCompatibleEvaluationCandidates } from '../utils/evaluationComparison'
import {
    formatCreatedAgo,
    formatDateTime,
    formatRemainingTime,
    historySummary,
    isNearExpiration,
} from '../utils/evaluationHistory'
import {
    individualAnalysisOutcome,
    individualAnalysisSummary,
} from '../utils/individualAnalysisHistory'

const HISTORY_PAGE_SIZE = 8
const COMPARISON_CANDIDATE_STEP = 8

function ArrowIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 12h13M14 7l5 5-5 5" />
        </svg>
    )
}

function CompareIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 7h11M13 4l3 3-3 3M19 17H8M11 14l-3 3 3 3" />
        </svg>
    )
}

function MoreIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="5" cy="12" r="1.2" />
            <circle cx="12" cy="12" r="1.2" />
            <circle cx="19" cy="12" r="1.2" />
        </svg>
    )
}

function PinIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m8 4 8 0-1.5 5 2.5 2.5v1.5H7v-1.5L9.5 9 8 4ZM12 13v7" />
        </svg>
    )
}

function TrashIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 7h14M9 7V4h6v3M8 10v8M12 10v8M16 10v8M6.5 7l.8 13h9.4l.8-13" />
        </svg>
    )
}

function HistoryStatus({ item }) {
    if (item.status === 'running') {
        return <span className="dashboard-history-status running">En curso</span>
    }
    if (item.preserved) {
        return <span className="dashboard-history-status preserved">Conservada</span>
    }
    if (isNearExpiration(item)) {
        return <span className="dashboard-history-status expiring">Próxima a caducar</span>
    }
    return <span className="dashboard-history-status completed">Completada</span>
}

function TaskLabel({ item }) {
    return (
        <span className={`dashboard-task-label ${item.task_type}`}>
            {item.task_type === 'detection' ? 'Detección' : 'Clasificación'}
        </span>
    )
}

function EvaluationCopy({ item, attackCatalog }) {
    return (
        <>
            <div className="dashboard-history-identity">
                <TaskLabel item={item} />
                <div>
                    <strong title={item.dataset_name}>{item.dataset_name}</strong>
                    <small>{item.model_name}</small>
                </div>
            </div>

            <div className="dashboard-history-config">
                <span>{getAttackName(item.attack_type, attackCatalog)}</span>
                <small
                    title={formatAttackParameters(
                        item.attack_type,
                        item.parameters,
                        attackCatalog,
                    )}
                >
                    {formatAttackParameters(
                        item.attack_type,
                        item.parameters,
                        attackCatalog,
                    )}
                </small>
            </div>

            <div className="dashboard-history-result">
                <strong>{historySummary(item)}</strong>
                <small title={formatDateTime(item.created_at)}>
                    {formatCreatedAgo(item.created_at)}
                    {item.preserved
                        ? ' · conservación indefinida'
                        : item.status !== 'running' && item.expires_at
                            ? ` · caduca en ${formatRemainingTime(item.expires_at)}`
                            : ''}
                </small>
            </div>
        </>
    )
}

function findAnalysisImage(datasets, item) {
    const dataset = datasets.find((candidate) => candidate.id === item.dataset_id)
    const image = dataset?.images?.find((candidate) => candidate.id === item.image_id)
    return { dataset, image }
}

function IndividualAnalysisCopy({ item, datasets, attackCatalog }) {
    const { image } = findAnalysisImage(datasets, item)
    const imageUrl = resolveApiUrl(image?.url)

    return (
        <>
            <div className="dashboard-individual-identity">
                <div className="dashboard-individual-thumbnail" aria-hidden="true">
                    {imageUrl ? (
                        <img src={imageUrl} alt="" />
                    ) : (
                        <span>IMG</span>
                    )}
                </div>
                <div className="dashboard-individual-identity-copy">
                    <div>
                        <TaskLabel item={item} />
                        {item.explanation_available && (
                            <span className="dashboard-analysis-artifact-badge">
                                {item.task_type === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'}
                            </span>
                        )}
                    </div>
                    <strong title={item.image_filename}>{item.image_filename}</strong>
                    <small title={item.dataset_name}>{item.dataset_name} · {item.model_name}</small>
                </div>
            </div>

            <div className="dashboard-history-config">
                <span>{getAttackName(item.attack_type, attackCatalog)}</span>
                <small
                    title={formatAttackParameters(
                        item.attack_type,
                        item.parameters,
                        attackCatalog,
                    )}
                >
                    {formatAttackParameters(
                        item.attack_type,
                        item.parameters,
                        attackCatalog,
                    )}
                </small>
            </div>

            <div className="dashboard-history-result dashboard-individual-result">
                <strong>{individualAnalysisSummary(item)}</strong>
                <small>
                    {individualAnalysisOutcome(item)} · guardado {formatCreatedAgo(item.saved_at)}
                </small>
            </div>
        </>
    )
}

function HistoryMoreMenu({ children, label = 'Más acciones' }) {
    return (
        <details className="dashboard-history-more">
            <summary aria-label={label} title={label}>
                <MoreIcon />
            </summary>
            <div className="dashboard-history-more-menu">
                {children}
            </div>
        </details>
    )
}

function HistoryConfirmationDialog({ confirmation, busy, onCancel, onConfirm }) {
    if (!confirmation) return null

    const isDelete = confirmation.type.startsWith('delete')
    const isIndividual = confirmation.type === 'delete-individual'
    const title = isDelete
        ? `Eliminar ${isIndividual ? 'análisis individual' : 'evaluación'}`
        : 'Dejar de conservar la evaluación'
    const description = isDelete
        ? isIndividual
            ? 'El análisis guardado y sus artefactos persistidos se eliminarán de forma permanente. El dataset y el modelo no se modificarán.'
            : 'La evaluación y sus artefactos almacenados se eliminarán de forma permanente. El dataset, el modelo y otras evaluaciones no se modificarán.'
        : 'La evaluación volverá a estar sujeta a tu política de retención actual y recibirá una nueva fecha de caducidad a partir de ahora.'

    return (
        <div
            className="dashboard-history-dialog-backdrop"
            role="presentation"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget && !busy) onCancel()
            }}
        >
            <section
                className="dashboard-history-dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="history-confirmation-title"
            >
                <span className="dashboard-section-eyebrow">
                    {isDelete ? 'Acción irreversible' : 'Retención'}
                </span>
                <h2 id="history-confirmation-title">{title}</h2>
                <p>{description}</p>
                <div className="dashboard-history-dialog-actions">
                    <button
                        type="button"
                        className="secondary-action"
                        disabled={busy}
                        onClick={onCancel}
                    >
                        Cancelar
                    </button>
                    <button
                        type="button"
                        className={isDelete ? 'danger-action' : 'primary-action'}
                        disabled={busy}
                        onClick={onConfirm}
                    >
                        {busy
                            ? isDelete ? 'Eliminando…' : 'Aplicando…'
                            : isDelete
                                ? 'Eliminar'
                                : 'Aplicar retención'}
                    </button>
                </div>
            </section>
        </div>
    )
}

function DashboardHistory({
    datasets,
    attackCatalog,
    openRecoveredEvaluation,
    openRecoveredIndividualAnalysis,
    onNavigateToEvaluation,
    onNavigateToIndividualAnalysis,
    onNavigateToProfile,
    onNavigateToComparison,
}) {
    const [historyKind, setHistoryKind] = useState('evaluations')
    const [history, setHistory] = useState({ items: [], total: 0 })
    const [offset, setOffset] = useState(0)
    const [taskType, setTaskType] = useState('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [openingResultId, setOpeningResultId] = useState('')
    const [mutationResultId, setMutationResultId] = useState('')
    const [refreshKey, setRefreshKey] = useState(0)
    const [confirmation, setConfirmation] = useState(null)
    const [comparisonBase, setComparisonBase] = useState(null)
    const [comparisonCandidates, setComparisonCandidates] = useState([])
    const [comparisonLoading, setComparisonLoading] = useState(false)
    const [comparisonVisibleCount, setComparisonVisibleCount] = useState(
        COMPARISON_CANDIDATE_STEP,
    )

    useEffect(() => {
        let cancelled = false

        async function loadHistory() {
            try {
                setLoading(true)
                setError('')
                const request = historyKind === 'evaluations'
                    ? getEvaluationHistory
                    : getIndividualAnalysisHistory
                const page = await request({
                    limit: HISTORY_PAGE_SIZE,
                    offset,
                    taskType,
                })
                if (!cancelled) setHistory(page)
            } catch (requestError) {
                if (!cancelled) setError(requestError.message)
            } finally {
                if (!cancelled) setLoading(false)
            }
        }

        loadHistory()
        return () => {
            cancelled = true
        }
    }, [historyKind, offset, taskType, refreshKey])

    function closeComparisonPicker() {
        setComparisonBase(null)
        setComparisonCandidates([])
        setComparisonLoading(false)
        setComparisonVisibleCount(COMPARISON_CANDIDATE_STEP)
    }

    function handleHistoryKind(value) {
        closeComparisonPicker()
        setHistoryKind(value)
        setTaskType('')
        setOffset(0)
        setError('')
    }

    function handleTaskFilter(value) {
        closeComparisonPicker()
        setTaskType(value)
        setOffset(0)
    }

    function refreshAfterDelete() {
        if (history.items.length === 1 && offset > 0) {
            setOffset(Math.max(0, offset - HISTORY_PAGE_SIZE))
        } else {
            setRefreshKey((current) => current + 1)
        }
    }

    async function handleOpenEvaluation(item) {
        const dataset = datasets.find((candidate) => candidate.id === item.dataset_id)
        if (!dataset) {
            setError(
                'El conjunto de datos asociado ya no está disponible y el resultado no puede reabrirse.',
            )
            return
        }

        try {
            setOpeningResultId(item.result_id)
            setError('')
            const stored = await getStoredEvaluation(item.result_id)
            openRecoveredEvaluation({ historyItem: item, stored, dataset })
            onNavigateToEvaluation()
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setOpeningResultId('')
        }
    }

    async function handleOpenIndividualAnalysis(item) {
        const { dataset, image } = findAnalysisImage(datasets, item)
        if (!dataset || !image) {
            setError(
                'La imagen original asociada ya no está disponible y el análisis no puede reabrirse.',
            )
            return
        }

        try {
            setOpeningResultId(item.analysis_id)
            setError('')
            const stored = await getStoredIndividualAnalysis(item.analysis_id)
            openRecoveredIndividualAnalysis({ historyItem: item, stored, dataset, image })
            onNavigateToIndividualAnalysis()
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setOpeningResultId('')
        }
    }

    async function handlePreserveEvaluation(item) {
        try {
            setMutationResultId(item.result_id)
            setError('')
            const updated = await setEvaluationPreserved(item.result_id, true)
            setHistory((current) => ({
                ...current,
                items: current.items.map((candidate) =>
                    candidate.result_id === updated.result_id ? updated : candidate),
            }))
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setMutationResultId('')
        }
    }

    async function handleConfirmMutation() {
        if (!confirmation) return

        const { type, item } = confirmation
        const id = type === 'delete-individual' ? item.analysis_id : item.result_id
        try {
            setMutationResultId(id)
            setError('')
            if (type === 'delete-evaluation') {
                await deleteEvaluation(item.result_id)
                setConfirmation(null)
                refreshAfterDelete()
            } else if (type === 'delete-individual') {
                await deleteIndividualAnalysis(item.analysis_id)
                setConfirmation(null)
                refreshAfterDelete()
            } else if (type === 'unpreserve-evaluation') {
                const updated = await setEvaluationPreserved(item.result_id, false)
                setHistory((current) => ({
                    ...current,
                    items: current.items.map((candidate) =>
                        candidate.result_id === updated.result_id ? updated : candidate),
                }))
                setConfirmation(null)
            }
        } catch (requestError) {
            setConfirmation(null)
            setError(requestError.message)
        } finally {
            setMutationResultId('')
        }
    }

    async function handleStartComparison(item) {
        if (item.status !== 'completed') return

        try {
            setComparisonBase(item)
            setComparisonCandidates([])
            setComparisonVisibleCount(COMPARISON_CANDIDATE_STEP)
            setComparisonLoading(true)
            setError('')
            const items = await getAllEvaluationHistory({ taskType: item.task_type })
            setComparisonCandidates(getCompatibleEvaluationCandidates(item, items))
        } catch (requestError) {
            setComparisonBase(null)
            setError(requestError.message)
        } finally {
            setComparisonLoading(false)
        }
    }

    const currentPage = Math.floor(offset / HISTORY_PAGE_SIZE) + 1
    const pageCount = Math.max(1, Math.ceil(history.total / HISTORY_PAGE_SIZE))
    const comparisonMode = historyKind === 'evaluations' && Boolean(comparisonBase)
    const isEvaluationHistory = historyKind === 'evaluations'

    return (
        <section className="dashboard-history-view">
            <header className="dashboard-history-header dashboard-unified-history-header">
                <div>
                    <span className="dashboard-section-eyebrow">Actividad conservada</span>
                    <h1>Historial</h1>
                    <p>
                        {isEvaluationHistory
                            ? 'Recupera evaluaciones anteriores, conserva resultados importantes o compara ejecuciones compatibles.'
                            : 'Reabre los análisis de una sola imagen que hayas decidido guardar explícitamente.'}
                    </p>
                </div>
                {isEvaluationHistory && (
                    <button
                        type="button"
                        className="dashboard-text-action"
                        onClick={onNavigateToProfile}
                    >
                        Gestionar retención automática
                    </button>
                )}
            </header>

            {!comparisonMode && (
                <nav className="dashboard-history-kinds" aria-label="Tipo de historial">
                    <button
                        type="button"
                        className={historyKind === 'evaluations' ? 'active' : ''}
                        onClick={() => handleHistoryKind('evaluations')}
                    >
                        <span>Evaluaciones</span>
                        <small>Dataset completo</small>
                    </button>
                    <button
                        type="button"
                        className={historyKind === 'individual' ? 'active' : ''}
                        onClick={() => handleHistoryKind('individual')}
                    >
                        <span>Análisis individuales</span>
                        <small>Una sola imagen</small>
                    </button>
                </nav>
            )}

            {!comparisonMode && (
                <div className="dashboard-history-toolbar">
                    <div className="dashboard-history-filters" aria-label="Filtrar historial">
                        {[
                            ['', 'Todas'],
                            ['classification', 'Clasificación'],
                            ['detection', 'Detección'],
                        ].map(([value, label]) => (
                            <button
                                key={value || 'all'}
                                type="button"
                                className={taskType === value ? 'active' : ''}
                                onClick={() => handleTaskFilter(value)}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                    <span>
                        {history.total}{' '}
                        {isEvaluationHistory ? 'evaluaciones disponibles' : 'análisis guardados'}
                    </span>
                </div>
            )}

            {error && <p className="error dashboard-history-error">{error}</p>}

            {comparisonMode ? (
                <div className="dashboard-comparison-picker">
                    <header className="dashboard-comparison-picker-heading">
                        <div>
                            <span className="dashboard-section-eyebrow">Comparación controlada</span>
                            <h2>Selecciona la evaluación B</h2>
                            <p>
                                Solo mostramos resultados donde cambia una única variable:
                                modelo, ataque o configuración del ataque.
                            </p>
                        </div>
                        <button
                            type="button"
                            className="dashboard-text-action"
                            onClick={closeComparisonPicker}
                        >
                            Cancelar
                        </button>
                    </header>

                    <article className="dashboard-comparison-base" aria-label="Evaluación A seleccionada">
                        <span className="dashboard-comparison-side side-a">A</span>
                        <div className="dashboard-comparison-base-copy">
                            <strong>{comparisonBase.model_name}</strong>
                            <span>
                                {getAttackName(comparisonBase.attack_type, attackCatalog)} ·{' '}
                                {formatAttackParameters(
                                    comparisonBase.attack_type,
                                    comparisonBase.parameters,
                                    attackCatalog,
                                )}
                            </span>
                            <small>
                                {comparisonBase.dataset_name} · revisión{' '}
                                {comparisonBase.dataset_revision}
                            </small>
                        </div>
                    </article>

                    {comparisonLoading ? (
                        <div className="dashboard-empty-state dashboard-history-loading">
                            <strong>Buscando comparaciones válidas…</strong>
                            <span>Comprobando el historial completo de esta tarea.</span>
                        </div>
                    ) : comparisonCandidates.length === 0 ? (
                        <div className="dashboard-empty-state dashboard-comparison-empty">
                            <strong>No hay otra evaluación compatible</strong>
                            <span>
                                Necesitas otra ejecución sobre la misma revisión del conjunto de
                                datos donde cambie solo el modelo, el ataque o su configuración.
                            </span>
                        </div>
                    ) : (
                        <div className="dashboard-comparison-candidates">
                            {comparisonCandidates
                                .slice(0, comparisonVisibleCount)
                                .map(({ item, comparison }) => (
                                    <article
                                        key={item.result_id}
                                        className="dashboard-comparison-candidate"
                                    >
                                        <span className="dashboard-comparison-side side-b">B</span>
                                        <EvaluationCopy item={item} attackCatalog={attackCatalog} />
                                        <div className="dashboard-comparison-candidate-action">
                                            <span className={`dashboard-comparison-kind kind-${comparison.kind}`}>
                                                {comparison.label}
                                            </span>
                                            <button
                                                type="button"
                                                className="dashboard-history-compare primary"
                                                onClick={() => onNavigateToComparison(comparisonBase, item)}
                                            >
                                                Comparar
                                                <ArrowIcon />
                                            </button>
                                        </div>
                                    </article>
                                ))}
                        </div>
                    )}

                    {!comparisonLoading
                    && comparisonCandidates.length > COMPARISON_CANDIDATE_STEP && (
                        <footer className="dashboard-comparison-candidate-footer">
                            <span>
                                {Math.min(comparisonVisibleCount, comparisonCandidates.length)} de{' '}
                                {comparisonCandidates.length} compatibles
                            </span>
                            <div>
                                {comparisonVisibleCount > COMPARISON_CANDIDATE_STEP && (
                                    <button
                                        type="button"
                                        onClick={() => setComparisonVisibleCount(COMPARISON_CANDIDATE_STEP)}
                                    >
                                        Mostrar menos
                                    </button>
                                )}
                                {comparisonVisibleCount < comparisonCandidates.length && (
                                    <button
                                        type="button"
                                        onClick={() => setComparisonVisibleCount((current) =>
                                            Math.min(
                                                current + COMPARISON_CANDIDATE_STEP,
                                                comparisonCandidates.length,
                                            ))}
                                    >
                                        Ver {Math.min(
                                            COMPARISON_CANDIDATE_STEP,
                                            comparisonCandidates.length - comparisonVisibleCount,
                                        )} más
                                    </button>
                                )}
                            </div>
                        </footer>
                    )}
                </div>
            ) : loading ? (
                <div className="dashboard-empty-state dashboard-history-loading">
                    <strong>Cargando historial…</strong>
                    <span>
                        {isEvaluationHistory
                            ? 'Recuperando tus evaluaciones vigentes.'
                            : 'Recuperando los análisis que has guardado.'}
                    </span>
                </div>
            ) : history.items.length === 0 ? (
                <div className="dashboard-empty-state dashboard-history-empty">
                    <strong>
                        {isEvaluationHistory
                            ? 'Aún no hay evaluaciones recuperables'
                            : 'Todavía no has guardado ningún análisis individual'}
                    </strong>
                    <span>
                        {isEvaluationHistory
                            ? 'Las evaluaciones aparecerán aquí mientras sigan vigentes o cuando decidas conservarlas.'
                            : 'Los análisis de imagen seguirán siendo temporales hasta que utilices Guardar análisis desde su vista.'}
                    </span>
                </div>
            ) : isEvaluationHistory ? (
                <div className="dashboard-history-list">
                    {history.items.map((item) => (
                        <article key={item.result_id} className="dashboard-history-row">
                            <EvaluationCopy item={item} attackCatalog={attackCatalog} />

                            <div className="dashboard-history-actions">
                                <HistoryStatus item={item} />
                                <button
                                    type="button"
                                    className="dashboard-history-compare"
                                    disabled={item.status !== 'completed'}
                                    onClick={() => handleStartComparison(item)}
                                    title="Seleccionar como evaluación A"
                                >
                                    <CompareIcon />
                                    <span>Comparar</span>
                                </button>
                                <button
                                    type="button"
                                    className="dashboard-history-open"
                                    disabled={
                                        item.status !== 'completed'
                                        || openingResultId === item.result_id
                                    }
                                    onClick={() => handleOpenEvaluation(item)}
                                >
                                    <span>
                                        {openingResultId === item.result_id ? 'Abriendo…' : 'Abrir'}
                                    </span>
                                    <ArrowIcon />
                                </button>
                                {item.status === 'completed' && (
                                    <HistoryMoreMenu label={`Más acciones para ${item.dataset_name}`}>
                                        <button
                                            type="button"
                                            disabled={mutationResultId === item.result_id}
                                            onClick={(event) => {
                                                event.currentTarget.closest('details')?.removeAttribute('open')
                                                if (item.preserved) {
                                                    setConfirmation({
                                                        type: 'unpreserve-evaluation',
                                                        item,
                                                    })
                                                } else {
                                                    handlePreserveEvaluation(item)
                                                }
                                            }}
                                        >
                                            <PinIcon />
                                            <span>
                                                {item.preserved
                                                    ? 'Volver a retención automática'
                                                    : 'Conservar indefinidamente'}
                                            </span>
                                        </button>
                                        <button
                                            type="button"
                                            className="danger"
                                            disabled={mutationResultId === item.result_id}
                                            onClick={(event) => {
                                                event.currentTarget.closest('details')?.removeAttribute('open')
                                                setConfirmation({ type: 'delete-evaluation', item })
                                            }}
                                        >
                                            <TrashIcon />
                                            <span>Eliminar evaluación</span>
                                        </button>
                                    </HistoryMoreMenu>
                                )}
                            </div>
                        </article>
                    ))}
                </div>
            ) : (
                <div className="dashboard-history-list dashboard-individual-history-list">
                    {history.items.map((item) => (
                        <article
                            key={item.analysis_id}
                            className="dashboard-history-row dashboard-individual-history-row"
                        >
                            <IndividualAnalysisCopy
                                item={item}
                                datasets={datasets}
                                attackCatalog={attackCatalog}
                            />

                            <div className="dashboard-history-actions">
                                <span className="dashboard-history-status preserved">Conservado</span>
                                <button
                                    type="button"
                                    className="dashboard-history-open"
                                    disabled={openingResultId === item.analysis_id}
                                    onClick={() => handleOpenIndividualAnalysis(item)}
                                >
                                    <span>
                                        {openingResultId === item.analysis_id ? 'Abriendo…' : 'Abrir'}
                                    </span>
                                    <ArrowIcon />
                                </button>
                                <HistoryMoreMenu label={`Más acciones para ${item.image_filename}`}>
                                    <button
                                        type="button"
                                        className="danger"
                                        disabled={mutationResultId === item.analysis_id}
                                        onClick={(event) => {
                                            event.currentTarget.closest('details')?.removeAttribute('open')
                                            setConfirmation({ type: 'delete-individual', item })
                                        }}
                                    >
                                        <TrashIcon />
                                        <span>Eliminar análisis</span>
                                    </button>
                                </HistoryMoreMenu>
                            </div>
                        </article>
                    ))}
                </div>
            )}

            {!comparisonMode && history.total > HISTORY_PAGE_SIZE && (
                <footer className="dashboard-history-pagination">
                    <span>Página {currentPage} de {pageCount}</span>
                    <div>
                        <button
                            type="button"
                            disabled={offset === 0 || loading}
                            onClick={() => setOffset(Math.max(0, offset - HISTORY_PAGE_SIZE))}
                        >
                            Anterior
                        </button>
                        <button
                            type="button"
                            disabled={offset + HISTORY_PAGE_SIZE >= history.total || loading}
                            onClick={() => setOffset(offset + HISTORY_PAGE_SIZE)}
                        >
                            Siguiente
                        </button>
                    </div>
                </footer>
            )}

            <HistoryConfirmationDialog
                confirmation={confirmation}
                busy={Boolean(mutationResultId)}
                onCancel={() => {
                    if (!mutationResultId) setConfirmation(null)
                }}
                onConfirm={handleConfirmMutation}
            />
        </section>
    )
}

export default DashboardHistory
