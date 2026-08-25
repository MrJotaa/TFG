import { useEffect, useState } from 'react'
import {
    getEvaluationHistory,
    getStoredEvaluation,
} from '../api/evaluationHistory'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import {
    formatCreatedAgo,
    formatDateTime,
    formatRemainingTime,
    historySummary,
    isNearExpiration,
} from '../utils/evaluationHistory'

const HISTORY_PAGE_SIZE = 8

function ArrowIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 12h13M14 7l5 5-5 5" />
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

function DashboardHistory({
    datasets,
    attackCatalog,
    openRecoveredEvaluation,
    onNavigateToEvaluation,
    onNavigateToProfile,
}) {
    const [history, setHistory] = useState({ items: [], total: 0 })
    const [offset, setOffset] = useState(0)
    const [taskType, setTaskType] = useState('')
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')
    const [openingResultId, setOpeningResultId] = useState('')

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

    const currentPage = Math.floor(offset / HISTORY_PAGE_SIZE) + 1
    const pageCount = Math.max(1, Math.ceil(history.total / HISTORY_PAGE_SIZE))

    return (
        <section className="dashboard-history-view">
            <header className="dashboard-history-header">
                <div>
                    <span className="dashboard-section-eyebrow">Actividad reciente</span>
                    <h1>Historial de evaluaciones</h1>
                    <p>Recupera resultados anteriores sin volver a ejecutar el análisis.</p>
                </div>
                <button
                    type="button"
                    className="dashboard-text-action"
                    onClick={onNavigateToProfile}
                >
                    Gestionar retención
                </button>
            </header>

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
                            onClick={() => {
                                setTaskType(value)
                                setOffset(0)
                            }}
                        >
                            {label}
                        </button>
                    ))}
                </div>
                <span>{history.total} resultados vigentes</span>
            </div>

            {error && <p className="error dashboard-history-error">{error}</p>}

            {loading ? (
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
                            <div className="dashboard-history-identity">
                                <span className={`dashboard-task-label ${item.task_type}`}>
                                    {item.task_type === 'detection' ? 'Detección' : 'Clasificación'}
                                </span>
                                <div>
                                    <strong title={item.dataset_name}>{item.dataset_name}</strong>
                                    <small>{item.model_name}</small>
                                </div>
                            </div>

                            <div className="dashboard-history-config">
                                <span>{getAttackName(item.attack_type, attackCatalog)}</span>
                                <small title={formatAttackParameters(item.attack_type, item.parameters, attackCatalog)}>
                                    {formatAttackParameters(item.attack_type, item.parameters, attackCatalog)}
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

                            <div className="dashboard-history-actions">
                                <HistoryStatus item={item} />
                                <button
                                    type="button"
                                    className="dashboard-history-open"
                                    disabled={item.status !== 'completed' || openingResultId === item.result_id}
                                    onClick={() => handleOpenResult(item)}
                                >
                                    <span>{openingResultId === item.result_id ? 'Abriendo…' : 'Abrir'}</span>
                                    <ArrowIcon />
                                </button>
                            </div>
                        </article>
                    ))}
                </div>
            )}

            {history.total > HISTORY_PAGE_SIZE && (
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
