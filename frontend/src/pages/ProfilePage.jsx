import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    getEvaluationHistory,
    getStoredEvaluation,
} from '../api/evaluationHistory'
import { getDatasetsByUser } from '../api/datasets'
import {
    deleteCurrentUser,
    updateResultRetention,
    updateUser,
} from '../api/users'
import { useAuth } from '../context/AuthContext'
import { useExperiment } from '../context/ExperimentContext'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'
import {
    RESULT_RETENTION_OPTIONS,
    formatCreatedAgo,
    formatDateTime,
    formatRemainingTime,
    historySummary,
    isNearExpiration,
} from '../utils/evaluationHistory'

const HISTORY_PAGE_SIZE = 10

function HistoryStatus({ item }) {
    if (item.status === 'running') {
        return <span className="history-status running">En curso</span>
    }
    if (isNearExpiration(item)) {
        return <span className="history-status expiring">Próxima a caducar</span>
    }
    return <span className="history-status completed">Completada</span>
}

function ProfilePage() {
    const navigate = useNavigate()
    const { user, setUser, logout } = useAuth()
    const {
        attackCatalog,
        resetExperiment,
        openRecoveredEvaluation,
    } = useExperiment()

    const [name, setName] = useState(user?.name || '')
    const [email, setEmail] = useState(user?.email || '')
    const [password, setPassword] = useState('')
    const [datasets, setDatasets] = useState([])
    const [history, setHistory] = useState({ items: [], total: 0 })
    const [historyOffset, setHistoryOffset] = useState(0)
    const [historyTask, setHistoryTask] = useState('')
    const [historyLoading, setHistoryLoading] = useState(false)
    const [historyError, setHistoryError] = useState('')
    const [openingResultId, setOpeningResultId] = useState('')
    const [retentionSeconds, setRetentionSeconds] = useState(
        user?.result_retention_seconds ?? 86400,
    )
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState('')
    const [deletePassword, setDeletePassword] = useState('')

    useEffect(() => {
        async function loadDatasets() {
            if (!user) return
            try {
                setError('')
                setDatasets(await getDatasetsByUser(user.id))
            } catch (requestError) {
                setError(requestError.message)
            }
        }
        loadDatasets()
    }, [user])

    useEffect(() => {
        async function loadHistory() {
            if (!user) return
            try {
                setHistoryLoading(true)
                setHistoryError('')
                const page = await getEvaluationHistory({
                    limit: HISTORY_PAGE_SIZE,
                    offset: historyOffset,
                    taskType: historyTask,
                })
                setHistory(page)
            } catch (requestError) {
                setHistoryError(requestError.message)
            } finally {
                setHistoryLoading(false)
            }
        }
        loadHistory()
    }, [historyOffset, historyTask, user])

    async function handleUpdateProfile(event) {
        event.preventDefault()
        if (!user) return
        const updateData = { name, email }
        if (password.trim()) updateData.password = password
        try {
            setLoading(true)
            setError('')
            setSuccess('')
            const updatedUser = await updateUser(user.id, updateData)
            setUser(updatedUser)
            setPassword('')
            setSuccess('Perfil actualizado correctamente.')
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setLoading(false)
        }
    }

    async function handleRetentionUpdate() {
        try {
            setLoading(true)
            setError('')
            setSuccess('')
            const updatedUser = await updateResultRetention(retentionSeconds)
            setUser(updatedUser)
            setSuccess(
                'La nueva retención se aplicará a las próximas evaluaciones.',
            )
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setLoading(false)
        }
    }

    async function handleOpenResult(item) {
        const dataset = datasets.find((candidate) => candidate.id === item.dataset_id)
        if (!dataset) {
            setHistoryError(
                'El conjunto de datos asociado ya no está disponible y el resultado no puede reabrirse.',
            )
            return
        }
        try {
            setOpeningResultId(item.result_id)
            setHistoryError('')
            const stored = await getStoredEvaluation(item.result_id)
            openRecoveredEvaluation({ historyItem: item, stored, dataset })
            navigate('/evaluation')
        } catch (requestError) {
            setHistoryError(requestError.message)
        } finally {
            setOpeningResultId('')
        }
    }

    async function handleDeleteAccount() {
        if (!user) return
        if (!deletePassword.trim()) {
            setError('Debes introducir tu contraseña para eliminar la cuenta.')
            return
        }
        const confirmed = window.confirm(
            '¿Seguro que quieres eliminar tu cuenta? También se eliminarán tus conjuntos de datos y resultados. Esta acción no se puede deshacer.',
        )
        if (!confirmed) return
        try {
            setLoading(true)
            setError('')
            await deleteCurrentUser(deletePassword)
            logout()
            resetExperiment()
            navigate('/')
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setLoading(false)
        }
    }

    if (!user) {
        return (
            <section className="profile-page">
                <div className="empty-state">
                    <h2>No hay usuario autenticado</h2>
                </div>
            </section>
        )
    }

    const historyPage = Math.floor(historyOffset / HISTORY_PAGE_SIZE) + 1
    const historyPageCount = Math.max(
        1,
        Math.ceil(history.total / HISTORY_PAGE_SIZE),
    )

    return (
        <section className="profile-page">
            <div className="page-toolbar">
                <div>
                    <h2>Configuración de usuario</h2>
                    <p>Gestiona tu cuenta, conjuntos de datos y resultados temporales.</p>
                </div>
                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard')}
                >
                    Volver al panel principal
                </button>
            </div>

            {error && <p className="error">{error}</p>}
            {success && <p className="success-message">{success}</p>}

            <div className="profile-layout">
                <section className="management-card profile-summary-card">
                    <h3>Datos del usuario</h3>
                    <form onSubmit={handleUpdateProfile} className="profile-form">
                        <div className="profile-form-grid">
                            <label>
                                Nombre
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(event) => setName(event.target.value)}
                                    required
                                />
                            </label>
                            <label>
                                Correo
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(event) => setEmail(event.target.value)}
                                    required
                                />
                            </label>
                        </div>
                        <label>
                            Nueva contraseña
                            <input
                                type="password"
                                value={password}
                                onChange={(event) => setPassword(event.target.value)}
                                placeholder="Dejar vacío para no cambiar"
                            />
                        </label>
                        <button type="submit" className="primary-action" disabled={loading}>
                            {loading ? 'Guardando…' : 'Guardar cambios'}
                        </button>
                    </form>
                </section>

                <section className="management-card profile-summary-card">
                    <div className="profile-card-heading">
                        <div>
                            <h3>Conjuntos de datos</h3>
                            <p><strong>{datasets.length}</strong> disponibles</p>
                        </div>
                        <button
                            type="button"
                            className="secondary-action small-inline"
                            onClick={() => navigate('/datasets')}
                        >
                            Gestionar
                        </button>
                    </div>
                    {datasets.length === 0 ? (
                        <p>No tienes conjuntos de datos creados.</p>
                    ) : (
                        <div className="profile-dataset-list compact">
                            {datasets.slice(0, 5).map((dataset) => (
                                <div key={dataset.id} className="profile-dataset-item">
                                    <strong>{dataset.name}</strong>
                                    <span>{dataset.image_count} imágenes</span>
                                </div>
                            ))}
                        </div>
                    )}
                </section>

                <section className="management-card profile-history-card">
                    <header className="profile-history-heading">
                        <div>
                            <span className="evaluation-eyebrow">Resultados temporales</span>
                            <h3>Historial de evaluaciones</h3>
                            <p>
                                Recupera resultados sin repetir el ataque. Cambiar la
                                retención no modifica los vencimientos existentes.
                            </p>
                        </div>
                        <div className="retention-control">
                            <label htmlFor="result-retention">Conservar resultados nuevos</label>
                            <div>
                                <select
                                    id="result-retention"
                                    value={retentionSeconds}
                                    onChange={(event) => setRetentionSeconds(Number(event.target.value))}
                                >
                                    {RESULT_RETENTION_OPTIONS.map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.label}
                                        </option>
                                    ))}
                                </select>
                                <button
                                    type="button"
                                    className="secondary-action"
                                    disabled={loading || retentionSeconds === user.result_retention_seconds}
                                    onClick={handleRetentionUpdate}
                                >
                                    Aplicar
                                </button>
                            </div>
                        </div>
                    </header>

                    <div className="history-toolbar">
                        <div className="history-task-filter" aria-label="Filtrar historial">
                            {[
                                ['', 'Todas'],
                                ['classification', 'Clasificación'],
                                ['detection', 'Detección'],
                            ].map(([value, label]) => (
                                <button
                                    key={value || 'all'}
                                    type="button"
                                    className={historyTask === value ? 'active' : ''}
                                    onClick={() => {
                                        setHistoryTask(value)
                                        setHistoryOffset(0)
                                    }}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                        <span>{history.total} resultados vigentes</span>
                    </div>

                    {historyError && <p className="error">{historyError}</p>}
                    {historyLoading ? (
                        <p className="history-empty">Cargando historial…</p>
                    ) : history.items.length === 0 ? (
                        <p className="history-empty">Aún no hay evaluaciones recuperables.</p>
                    ) : (
                        <div className="profile-history-table-wrapper">
                            <table className="profile-history-table">
                                <thead>
                                    <tr>
                                        <th>Evaluación</th>
                                        <th>Configuración</th>
                                        <th>Resultado</th>
                                        <th>Creada</th>
                                        <th>Caducidad</th>
                                        <th>Estado</th>
                                        <th><span className="evaluation-sr-only">Acciones</span></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {history.items.map((item) => (
                                        <tr key={item.result_id}>
                                            <td data-label="Evaluación">
                                                <span className={`task-pill ${item.task_type}`}>
                                                    {item.task_type === 'detection' ? 'Detección' : 'Clasificación'}
                                                </span>
                                                <strong title={item.dataset_name}>{item.dataset_name}</strong>
                                                <small>{item.model_name}</small>
                                            </td>
                                            <td data-label="Configuración">
                                                <strong>{getAttackName(item.attack_type, attackCatalog)}</strong>
                                                <small>{formatAttackParameters(item.attack_type, item.parameters, attackCatalog)}</small>
                                            </td>
                                            <td data-label="Resultado">{historySummary(item)}</td>
                                            <td data-label="Creada" title={formatDateTime(item.created_at)}>
                                                {formatCreatedAgo(item.created_at)}
                                            </td>
                                            <td data-label="Caducidad" title={formatDateTime(item.expires_at)}>
                                                {item.status === 'running' ? 'Al finalizar' : formatRemainingTime(item.expires_at)}
                                            </td>
                                            <td data-label="Estado"><HistoryStatus item={item} /></td>
                                            <td className="history-open-cell">
                                                <button
                                                    type="button"
                                                    className="secondary-action small-inline"
                                                    disabled={item.status !== 'completed' || openingResultId === item.result_id}
                                                    onClick={() => handleOpenResult(item)}
                                                >
                                                    {openingResultId === item.result_id ? 'Abriendo…' : 'Abrir'}
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    <footer className="history-pagination">
                        <span>Página {historyPage} de {historyPageCount}</span>
                        <div>
                            <button
                                type="button"
                                className="secondary-action small-inline"
                                disabled={historyOffset === 0 || historyLoading}
                                onClick={() => setHistoryOffset(Math.max(0, historyOffset - HISTORY_PAGE_SIZE))}
                            >
                                Anterior
                            </button>
                            <button
                                type="button"
                                className="secondary-action small-inline"
                                disabled={historyOffset + HISTORY_PAGE_SIZE >= history.total || historyLoading}
                                onClick={() => setHistoryOffset(historyOffset + HISTORY_PAGE_SIZE)}
                            >
                                Siguiente
                            </button>
                        </div>
                    </footer>
                </section>

                <section className="management-card danger-zone profile-danger-row">
                    <div>
                        <h3>Zona peligrosa</h3>
                        <p>Eliminar la cuenta borrará conjuntos de datos, evaluaciones y artefactos asociados.</p>
                    </div>
                    <label>
                        Confirma tu contraseña
                        <input
                            type="password"
                            value={deletePassword}
                            onChange={(event) => setDeletePassword(event.target.value)}
                            placeholder="Introduce tu contraseña"
                        />
                    </label>
                    <button
                        type="button"
                        className="danger-action"
                        onClick={handleDeleteAccount}
                        disabled={loading}
                    >
                        Eliminar cuenta
                    </button>
                </section>
            </div>
        </section>
    )
}

export default ProfilePage
