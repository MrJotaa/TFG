import { useEffect, useState } from 'react'
import {
    getAllEvaluationHistory,
    getEvaluationHistory,
    getStoredEvaluation,
} from '../api/evaluationHistory'
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

function HistoryStatus({ item }) {
    if (item.status === 'running') {
        return <span className="dashboard-history-status running">En curso</span>
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
                    {item.status !== 'running' && item.expires_at
                        ? ` · caduca en ${formatRemainingTime(item.expires_at)}`
                        : ''}
                </small>
            </div>
        </>
    )
}

function DashboardHistory({
    datasets,
    attackCatalog,
    openRecoveredEvaluation,
    onNavigateToEvaluation,
    onNavigateToProfile,
    onNavigateToComparison,
}) {
    const [history, setHistory] = useState({ items: [], total: 0 })
    const [offset, setOffset] = useState(0)
    const [taskType, setTaskType] = useState('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [openingResultId, setOpeningResultId] = useState('')
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
                const page = await getEvaluationHistory({
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
    }, [offset, taskType])

    async function handleOpenResult(item) {
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

    function closeComparisonPicker() {
        setComparisonBase(null)
        setComparisonCandidates([])
        setComparisonLoading(false)
        setComparisonVisibleCount(COMPARISON_CANDIDATE_STEP)
    }

    function handleTaskFilter(value) {
        closeComparisonPicker()
        setTaskType(value)
        setOffset(0)
    }

    const currentPage = Math.floor(offset / HISTORY_PAGE_SIZE) + 1
    const pageCount = Math.max(1, Math.ceil(history.total / HISTORY_PAGE_SIZE))
    const comparisonMode = Boolean(comparisonBase)

    return (
        <section className="dashboard-history-view">
            <header className="dashboard-history-header">
                <div>
                    <span className="dashboard-section-eyebrow">Actividad reciente</span>
                    <h1>Historial de evaluaciones</h1>
                    <p>
                        Recupera resultados anteriores o compara dos evaluaciones compatibles
                        sin volver a ejecutar el análisis.
                    </p>
                </div>
                <button
                    type="button"
                    className="dashboard-text-action"
                    onClick={onNavigateToProfile}
                >
                    Gestionar retención
                </button>
            </header>

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
                    <span>{history.total} resultados vigentes</span>
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
                    <span>Recuperando tus evaluaciones vigentes.</span>
                </div>
            ) : history.items.length === 0 ? (
                <div className="dashboard-empty-state dashboard-history-empty">
                    <strong>Aún no hay evaluaciones recuperables</strong>
                    <span>Las evaluaciones completadas aparecerán aquí mientras sigan vigentes.</span>
                </div>
            ) : (
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
                                    onClick={() => handleOpenResult(item)}
                                >
                                    <span>
                                        {openingResultId === item.result_id ? 'Abriendo…' : 'Abrir'}
                                    </span>
                                    <ArrowIcon />
                                </button>
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
        </section>
    )
}

export default DashboardHistory
