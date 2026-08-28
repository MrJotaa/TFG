import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    deleteEvaluation,
    getEvaluationHistory,
    getStoredEvaluation,
    setEvaluationPreserved,
} from '../api/evaluationHistory'
import { getDatasetsByUser } from '../api/datasets'
import {
    deleteIndividualAnalysis,
    getIndividualAnalysisHistory,
    getStoredIndividualAnalysis,
} from '../api/individualAnalysisHistory'
import {
    deleteCurrentUser,
    updateResultRetention,
    updateUser,
} from '../api/users'
import {
    HistoryConfirmationDialog,
    HistoryMoreMenu,
    PinIcon,
    TrashIcon,
} from '../components/DashboardHistory'
import PasswordField from '../components/PasswordField'
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
import {
    individualAnalysisOutcome,
    individualAnalysisSummary,
} from '../utils/individualAnalysisHistory'
import {
    buildProfileUpdatePayload,
    getProfilePasswordErrors,
    hasPasswordErrors,
    hasProfileChanges,
    wantsPasswordChange,
} from '../utils/profileForms'

const HISTORY_PAGE_SIZE = 10

function HistoryStatus({ item }) {
    if (item.status === 'running') {
        return <span className="history-status running">En curso</span>
    }
    if (item.preserved) {
        return <span className="history-status preserved">Conservada</span>
    }
    if (isNearExpiration(item)) {
        return <span className="history-status expiring">Próxima a caducar</span>
    }
    return <span className="history-status completed">Completada</span>
}

function DangerIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3.5 21 19H3L12 3.5Z" />
            <path d="M12 9v4.8" />
            <path d="M12 17.2h.01" />
        </svg>
    )
}

function ProfilePage() {
    const navigate = useNavigate()
    const { user, setUser, logout } = useAuth()
    const {
        attackCatalog,
        resetExperiment,
        openRecoveredEvaluation,
        openRecoveredIndividualAnalysis,
    } = useExperiment()

    const [name, setName] = useState(user?.name || '')
    const [email, setEmail] = useState(user?.email || '')
    const [showPasswordChange, setShowPasswordChange] = useState(false)
    const [currentPassword, setCurrentPassword] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [confirmationTouched, setConfirmationTouched] = useState(false)
    const [profileAttempted, setProfileAttempted] = useState(false)
    const [currentPasswordVisible, setCurrentPasswordVisible] = useState(false)
    const [newPasswordsVisible, setNewPasswordsVisible] = useState(false)
    const [profileSaving, setProfileSaving] = useState(false)
    const [profileError, setProfileError] = useState('')
    const [profileSuccess, setProfileSuccess] = useState('')

    const [datasets, setDatasets] = useState([])
    const [datasetError, setDatasetError] = useState('')

    const [history, setHistory] = useState({ items: [], total: 0 })
    const [historyKind, setHistoryKind] = useState('evaluations')
    const [historyOffset, setHistoryOffset] = useState(0)
    const [historyTask, setHistoryTask] = useState('')
    const [historyLoading, setHistoryLoading] = useState(false)
    const [historyError, setHistoryError] = useState('')
    const [openingResultId, setOpeningResultId] = useState('')
    const [historyMutationId, setHistoryMutationId] = useState('')
    const [historyRefreshKey, setHistoryRefreshKey] = useState(0)
    const [historyConfirmation, setHistoryConfirmation] = useState(null)
    const [historyOpenMenuId, setHistoryOpenMenuId] = useState(null)
    const [retentionSeconds, setRetentionSeconds] = useState(
        user?.result_retention_seconds ?? 86400,
    )
    const [retentionSaving, setRetentionSaving] = useState(false)
    const [retentionFeedback, setRetentionFeedback] = useState(null)

    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
    const [deletePassword, setDeletePassword] = useState('')
    const [deletePasswordVisible, setDeletePasswordVisible] = useState(false)
    const [deleteError, setDeleteError] = useState('')
    const [deletingAccount, setDeletingAccount] = useState(false)

    useEffect(() => {
        async function loadDatasets() {
            if (!user) return
            try {
                setDatasetError('')
                setDatasets(await getDatasetsByUser(user.id))
            } catch (requestError) {
                setDatasetError(requestError.message)
            }
        }
        loadDatasets()
    }, [user])

    useEffect(() => {
        let cancelled = false

        async function loadHistory() {
            if (!user) return
            try {
                setHistoryLoading(true)
                setHistoryError('')
                const request = historyKind === 'evaluations'
                    ? getEvaluationHistory
                    : getIndividualAnalysisHistory
                const page = await request({
                    limit: HISTORY_PAGE_SIZE,
                    offset: historyOffset,
                    taskType: historyTask,
                })
                if (!cancelled) setHistory(page)
            } catch (requestError) {
                if (!cancelled) setHistoryError(requestError.message)
            } finally {
                if (!cancelled) setHistoryLoading(false)
            }
        }
        loadHistory()
        return () => {
            cancelled = true
        }
    }, [historyKind, historyOffset, historyTask, historyRefreshKey, user])

    useEffect(() => {
        if (!historyOpenMenuId) return undefined

        function handlePointerDown(event) {
            if (!event.target.closest('.dashboard-history-more')) {
                setHistoryOpenMenuId(null)
            }
        }

        function handleKeyDown(event) {
            if (event.key === 'Escape') setHistoryOpenMenuId(null)
        }

        document.addEventListener('pointerdown', handlePointerDown)
        document.addEventListener('keydown', handleKeyDown)
        return () => {
            document.removeEventListener('pointerdown', handlePointerDown)
            document.removeEventListener('keydown', handleKeyDown)
        }
    }, [historyOpenMenuId])

    useEffect(() => {
        if (!historyConfirmation || historyMutationId) return undefined

        function handleKeyDown(event) {
            if (event.key === 'Escape') setHistoryConfirmation(null)
        }

        document.addEventListener('keydown', handleKeyDown)
        return () => document.removeEventListener('keydown', handleKeyDown)
    }, [historyConfirmation, historyMutationId])

    useEffect(() => {
        if (!deleteDialogOpen) return undefined

        function handleKeyDown(event) {
            if (event.key === 'Escape' && !deletingAccount) {
                setDeleteDialogOpen(false)
                setDeletePassword('')
                setDeleteError('')
            }
        }

        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [deleteDialogOpen, deletingAccount])

    const passwordFields = useMemo(
        () => ({ currentPassword, newPassword, confirmPassword }),
        [currentPassword, newPassword, confirmPassword],
    )
    const passwordChangeStarted = wantsPasswordChange(passwordFields)
    const passwordErrors = getProfilePasswordErrors({
        ...passwordFields,
        attempted: profileAttempted || passwordChangeStarted,
        confirmationTouched,
    })
    const validationPasswordErrors = getProfilePasswordErrors({
        ...passwordFields,
        attempted: true,
        confirmationTouched: true,
    })
    const profileChanged = hasProfileChanges({
        user,
        name,
        email,
        passwordFields,
    })
    const canSaveProfile = Boolean(
        name.trim() &&
            email.trim() &&
            profileChanged &&
            !hasPasswordErrors(validationPasswordErrors) &&
            !profileSaving,
    )

    function clearProfileFeedback() {
        setProfileError('')
        setProfileSuccess('')
    }

    function resetPasswordChange() {
        setCurrentPassword('')
        setNewPassword('')
        setConfirmPassword('')
        setConfirmationTouched(false)
        setProfileAttempted(false)
        setCurrentPasswordVisible(false)
        setNewPasswordsVisible(false)
    }

    function closePasswordChange() {
        resetPasswordChange()
        setShowPasswordChange(false)
        clearProfileFeedback()
    }

    async function handleUpdateProfile(event) {
        event.preventDefault()
        if (!user) return

        setProfileAttempted(true)
        const errors = getProfilePasswordErrors({
            ...passwordFields,
            attempted: true,
            confirmationTouched: true,
        })
        if (hasPasswordErrors(errors) || !name.trim() || !email.trim()) return

        const updateData = buildProfileUpdatePayload({
            user,
            name,
            email,
            currentPassword,
            newPassword,
            confirmPassword,
        })
        if (Object.keys(updateData).length === 0) return

        try {
            setProfileSaving(true)
            setProfileError('')
            setProfileSuccess('')
            const updatedUser = await updateUser(user.id, updateData)
            setUser(updatedUser)
            setName(updatedUser.name)
            setEmail(updatedUser.email)
            resetPasswordChange()
            setShowPasswordChange(false)
            setProfileSuccess('Datos de la cuenta actualizados correctamente.')
        } catch (requestError) {
            setProfileError(requestError.message)
        } finally {
            setProfileSaving(false)
        }
    }

    async function handleRetentionUpdate() {
        try {
            setRetentionSaving(true)
            setRetentionFeedback(null)
            const updatedUser = await updateResultRetention(retentionSeconds)
            setUser(updatedUser)
            setRetentionFeedback({
                type: 'success',
                message: 'La nueva retención se aplicará a las próximas evaluaciones.',
            })
        } catch (requestError) {
            setRetentionFeedback({ type: 'error', message: requestError.message })
        } finally {
            setRetentionSaving(false)
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

    async function handleOpenIndividualAnalysis(item) {
        const dataset = datasets.find((candidate) => candidate.id === item.dataset_id)
        const image = dataset?.images?.find((candidate) => candidate.id === item.image_id)
        if (!dataset || !image) {
            setHistoryError(
                'La imagen original asociada ya no está disponible y el análisis no puede reabrirse.',
            )
            return
        }
        try {
            setOpeningResultId(item.analysis_id)
            setHistoryError('')
            const stored = await getStoredIndividualAnalysis(item.analysis_id)
            openRecoveredIndividualAnalysis({ historyItem: item, stored, dataset, image })
            navigate('/comparison')
        } catch (requestError) {
            setHistoryError(requestError.message)
        } finally {
            setOpeningResultId('')
        }
    }

    function refreshHistoryAfterDelete() {
        if (history.items.length === 1 && historyOffset > 0) {
            setHistoryOffset(Math.max(0, historyOffset - HISTORY_PAGE_SIZE))
        } else {
            setHistoryRefreshKey((current) => current + 1)
        }
    }

    async function handlePreserveEvaluation(item) {
        try {
            setHistoryMutationId(item.result_id)
            setHistoryError('')
            const updated = await setEvaluationPreserved(item.result_id, true)
            setHistory((current) => ({
                ...current,
                items: current.items.map((candidate) =>
                    candidate.result_id === updated.result_id ? updated : candidate),
            }))
        } catch (requestError) {
            setHistoryError(requestError.message)
        } finally {
            setHistoryMutationId('')
        }
    }

    async function handleConfirmHistoryMutation() {
        if (!historyConfirmation) return

        const { type, item } = historyConfirmation
        const id = type === 'delete-individual' ? item.analysis_id : item.result_id
        try {
            setHistoryMutationId(id)
            setHistoryError('')
            if (type === 'delete-evaluation') {
                await deleteEvaluation(item.result_id)
                setHistoryConfirmation(null)
                refreshHistoryAfterDelete()
            } else if (type === 'delete-individual') {
                await deleteIndividualAnalysis(item.analysis_id)
                setHistoryConfirmation(null)
                refreshHistoryAfterDelete()
            } else if (type === 'unpreserve-evaluation') {
                const updated = await setEvaluationPreserved(item.result_id, false)
                setHistory((current) => ({
                    ...current,
                    items: current.items.map((candidate) =>
                        candidate.result_id === updated.result_id ? updated : candidate),
                }))
                setHistoryConfirmation(null)
            }
        } catch (requestError) {
            setHistoryConfirmation(null)
            setHistoryError(requestError.message)
        } finally {
            setHistoryMutationId('')
        }
    }

    function toggleHistoryMoreMenu(menuId) {
        setHistoryOpenMenuId((current) => current === menuId ? null : menuId)
    }

    function handleHistoryKind(value) {
        setHistoryOpenMenuId(null)
        setHistoryConfirmation(null)
        setHistoryKind(value)
        setHistory({ items: [], total: 0 })
        setHistoryTask('')
        setHistoryOffset(0)
        setHistoryError('')
    }

    function openDeleteDialog() {
        setDeletePassword('')
        setDeleteError('')
        setDeletePasswordVisible(false)
        setDeleteDialogOpen(true)
    }

    function closeDeleteDialog() {
        if (deletingAccount) return
        setDeleteDialogOpen(false)
        setDeletePassword('')
        setDeleteError('')
    }

    async function handleDeleteAccount(event) {
        event.preventDefault()
        if (!user || !deletePassword.trim()) {
            setDeleteError('Debes introducir tu contraseña para eliminar la cuenta.')
            return
        }

        try {
            setDeletingAccount(true)
            setDeleteError('')
            await deleteCurrentUser(deletePassword)
            logout()
            resetExperiment()
            navigate('/')
        } catch (requestError) {
            setDeleteError(requestError.message)
        } finally {
            setDeletingAccount(false)
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
    const hiddenDatasetCount = Math.max(0, datasets.length - 5)

    return (
        <section className="profile-page profile-settings-page">
            <header className="profile-page-heading">
                <span className="profile-eyebrow">Cuenta</span>
                <h1>Configuración de usuario</h1>
                <p>Gestiona tus datos, conjuntos de datos, preferencias e historial de resultados.</p>
            </header>

            <div className="profile-overview-grid">
                <section className="profile-section profile-account-section">
                    <header className="profile-section-heading">
                        <div>
                            <h2>Datos de la cuenta</h2>
                            <p>Actualiza tu identidad o protege el acceso con una nueva contraseña.</p>
                        </div>
                    </header>

                    <form onSubmit={handleUpdateProfile} className="profile-account-form">
                        <div className="profile-form-grid">
                            <label className="profile-field">
                                <span>Nombre</span>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(event) => {
                                        setName(event.target.value)
                                        clearProfileFeedback()
                                    }}
                                    autoComplete="name"
                                    required
                                    disabled={profileSaving}
                                />
                            </label>
                            <label className="profile-field">
                                <span>Correo</span>
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(event) => {
                                        setEmail(event.target.value)
                                        clearProfileFeedback()
                                    }}
                                    autoComplete="email"
                                    required
                                    disabled={profileSaving}
                                />
                            </label>
                        </div>

                        <div className="profile-password-toggle-row">
                            <div>
                                <strong>Contraseña</strong>
                                <span>Solo necesitas verificar tu contraseña actual si quieres cambiarla.</span>
                            </div>
                            <button
                                type="button"
                                className="profile-text-action"
                                onClick={() => {
                                    if (showPasswordChange) closePasswordChange()
                                    else {
                                        setShowPasswordChange(true)
                                        clearProfileFeedback()
                                    }
                                }}
                                disabled={profileSaving}
                                aria-expanded={showPasswordChange}
                            >
                                {showPasswordChange ? 'Cancelar cambio' : 'Cambiar contraseña'}
                            </button>
                        </div>

                        {showPasswordChange && (
                            <div className="profile-password-panel">
                                <PasswordField
                                    id="profile-current-password"
                                    label="Contraseña actual"
                                    value={currentPassword}
                                    onChange={(event) => {
                                        setCurrentPassword(event.target.value)
                                        clearProfileFeedback()
                                    }}
                                    autoComplete="current-password"
                                    placeholder="Introduce tu contraseña actual"
                                    error={passwordErrors.currentPassword}
                                    disabled={profileSaving}
                                    required={passwordChangeStarted}
                                    visible={currentPasswordVisible}
                                    onVisibleChange={setCurrentPasswordVisible}
                                    visibilityLabel="contraseña actual"
                                />
                                <PasswordField
                                    id="profile-new-password"
                                    label="Nueva contraseña"
                                    value={newPassword}
                                    onChange={(event) => {
                                        setNewPassword(event.target.value)
                                        clearProfileFeedback()
                                    }}
                                    autoComplete="new-password"
                                    placeholder="Introduce la nueva contraseña"
                                    error={passwordErrors.newPassword}
                                    disabled={profileSaving}
                                    required={passwordChangeStarted}
                                    visible={newPasswordsVisible}
                                    onVisibleChange={setNewPasswordsVisible}
                                    visibilityLabel="nuevas contraseñas"
                                />
                                <PasswordField
                                    id="profile-confirm-password"
                                    label="Confirmar nueva contraseña"
                                    value={confirmPassword}
                                    onChange={(event) => {
                                        setConfirmPassword(event.target.value)
                                        clearProfileFeedback()
                                    }}
                                    onBlur={() => setConfirmationTouched(true)}
                                    autoComplete="new-password"
                                    placeholder="Repite la nueva contraseña"
                                    error={passwordErrors.confirmPassword}
                                    disabled={profileSaving}
                                    required={passwordChangeStarted}
                                    visible={newPasswordsVisible}
                                    showToggle={false}
                                />
                                <p className="profile-security-note">
                                    La contraseña actual se valida en el servidor antes de realizar el cambio.
                                </p>
                            </div>
                        )}

                        {profileError && <p className="profile-inline-message error" role="alert">{profileError}</p>}
                        {profileSuccess && <p className="profile-inline-message success-message">{profileSuccess}</p>}

                        <div className="profile-form-actions">
                            <button
                                type="submit"
                                className="primary-action"
                                disabled={!canSaveProfile}
                            >
                                {profileSaving ? 'Guardando…' : 'Guardar cambios'}
                            </button>
                            {!profileChanged && (
                                <span>No hay cambios pendientes.</span>
                            )}
                        </div>
                    </form>
                </section>

                <section className="profile-section profile-datasets-section">
                    <header className="profile-section-heading profile-dataset-heading">
                        <div>
                            <h2>Conjuntos de datos</h2>
                            <p><strong>{datasets.length}</strong> disponibles</p>
                        </div>
                        <button
                            type="button"
                            className="secondary-action small-inline"
                            onClick={() => navigate('/datasets')}
                        >
                            Gestionar
                        </button>
                    </header>

                    {datasetError && <p className="profile-inline-message error">{datasetError}</p>}
                    {datasets.length === 0 ? (
                        <p className="profile-empty-copy">No tienes conjuntos de datos creados.</p>
                    ) : (
                        <div className="profile-dataset-list-clean">
                            {datasets.slice(0, 5).map((dataset) => (
                                <div key={dataset.id} className="profile-dataset-row">
                                    <strong>{dataset.name}</strong>
                                    <span>{dataset.image_count} imágenes</span>
                                </div>
                            ))}
                            {hiddenDatasetCount > 0 && (
                                <p className="profile-dataset-more">
                                    y {hiddenDatasetCount} conjunto{hiddenDatasetCount === 1 ? '' : 's'} más
                                </p>
                            )}
                        </div>
                    )}
                </section>
            </div>

            <section className="profile-section profile-history-section">
                <header className="profile-history-heading-modern">
                    <div>
                        <span className="evaluation-eyebrow">Resultados guardados</span>
                        <h2>Historial de resultados</h2>
                        <p>
                            Consulta evaluaciones completas y análisis individuales sin repetir el ataque.
                        </p>
                    </div>
                    {historyKind === 'evaluations' && (
                        <div className="retention-control profile-retention-control">
                            <label htmlFor="result-retention">Retención de nuevas evaluaciones</label>
                            <div>
                                <select
                                    id="result-retention"
                                    value={retentionSeconds}
                                    onChange={(event) => {
                                        setRetentionSeconds(Number(event.target.value))
                                        setRetentionFeedback(null)
                                    }}
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
                                    disabled={
                                        retentionSaving ||
                                        retentionSeconds === user.result_retention_seconds
                                    }
                                    onClick={handleRetentionUpdate}
                                >
                                    {retentionSaving ? 'Aplicando…' : 'Aplicar'}
                                </button>
                            </div>
                            <small>Solo afecta a evaluaciones creadas a partir de ahora.</small>
                            {retentionFeedback && (
                                <span className={`retention-feedback ${retentionFeedback.type}`}>
                                    {retentionFeedback.message}
                                </span>
                            )}
                        </div>
                    )}
                </header>

                <nav className="profile-history-kind-tabs" aria-label="Tipo de historial">
                    {[
                        ['evaluations', 'Evaluaciones'],
                        ['individual', 'Análisis individuales'],
                    ].map(([value, label]) => (
                        <button
                            key={value}
                            type="button"
                            className={historyKind === value ? 'active' : ''}
                            aria-pressed={historyKind === value}
                            onClick={() => handleHistoryKind(value)}
                        >
                            {label}
                        </button>
                    ))}
                </nav>

                <div className="history-toolbar profile-history-toolbar">
                    <nav className="profile-history-tabs" aria-label="Filtrar por tarea">
                        {[
                            ['', 'Todas'],
                            ['classification', 'Clasificación'],
                            ['detection', 'Detección'],
                        ].map(([value, label]) => (
                            <button
                                key={value || 'all'}
                                type="button"
                                className={historyTask === value ? 'active' : ''}
                                aria-pressed={historyTask === value}
                                onClick={() => {
                                    setHistoryTask(value)
                                    setHistoryOffset(0)
                                }}
                            >
                                {label}
                            </button>
                        ))}
                    </nav>
                    <span>
                        {historyKind === 'evaluations'
                            ? `${history.total} ${history.total === 1 ? 'evaluación vigente' : 'evaluaciones vigentes'}`
                            : `${history.total} ${history.total === 1 ? 'análisis guardado' : 'análisis guardados'}`}
                    </span>
                </div>

                {historyError && <p className="profile-inline-message error">{historyError}</p>}
                {historyLoading ? (
                    <p className="history-empty">Cargando historial…</p>
                ) : history.items.length === 0 ? (
                    <p className="history-empty">
                        {historyKind === 'evaluations'
                            ? 'Aún no hay evaluaciones recuperables con este filtro.'
                            : 'Aún no hay análisis individuales guardados con este filtro.'}
                    </p>
                ) : historyKind === 'evaluations' ? (
                    <div className="profile-history-table-wrapper profile-history-table-modern">
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
                                            <span className={`dashboard-task-label ${item.task_type}`}>
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
                                        <td
                                            data-label="Caducidad"
                                            title={item.preserved ? 'Conservación indefinida' : formatDateTime(item.expires_at)}
                                        >
                                            {item.status === 'running'
                                                ? 'Al finalizar'
                                                : item.preserved
                                                    ? 'Indefinida'
                                                    : formatRemainingTime(item.expires_at)}
                                        </td>
                                        <td data-label="Estado"><HistoryStatus item={item} /></td>
                                        <td className="history-open-cell profile-history-actions-cell">
                                            <button
                                                type="button"
                                                className="secondary-action small-inline"
                                                disabled={item.status !== 'completed' || openingResultId === item.result_id}
                                                onClick={() => handleOpenResult(item)}
                                            >
                                                {openingResultId === item.result_id ? 'Abriendo…' : 'Abrir'}
                                            </button>
                                            {item.status === 'completed' && (
                                                <HistoryMoreMenu
                                                    label={`Más acciones para ${item.dataset_name}`}
                                                    menuId={`evaluation-${item.result_id}`}
                                                    open={historyOpenMenuId === `evaluation-${item.result_id}`}
                                                    onToggle={toggleHistoryMoreMenu}
                                                >
                                                    <button
                                                        type="button"
                                                        disabled={historyMutationId === item.result_id}
                                                        onClick={() => {
                                                            setHistoryOpenMenuId(null)
                                                            if (item.preserved) {
                                                                setHistoryConfirmation({
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
                                                        disabled={historyMutationId === item.result_id}
                                                        onClick={() => {
                                                            setHistoryOpenMenuId(null)
                                                            setHistoryConfirmation({
                                                                type: 'delete-evaluation',
                                                                item,
                                                            })
                                                        }}
                                                    >
                                                        <TrashIcon />
                                                        <span>Eliminar evaluación</span>
                                                    </button>
                                                </HistoryMoreMenu>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="profile-history-table-wrapper profile-history-table-modern">
                        <table className="profile-history-table profile-individual-history-table">
                            <thead>
                                <tr>
                                    <th>Análisis</th>
                                    <th>Configuración</th>
                                    <th>Resultado</th>
                                    <th>Guardado</th>
                                    <th>Estado</th>
                                    <th><span className="evaluation-sr-only">Acciones</span></th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.items.map((item) => (
                                    <tr key={item.analysis_id}>
                                        <td data-label="Análisis">
                                            <span className={`dashboard-task-label ${item.task_type}`}>
                                                {item.task_type === 'detection' ? 'Detección' : 'Clasificación'}
                                            </span>
                                            <strong title={item.image_filename}>{item.image_filename}</strong>
                                            <small title={item.dataset_name}>{item.dataset_name} · {item.model_name}</small>
                                        </td>
                                        <td data-label="Configuración">
                                            <strong>{getAttackName(item.attack_type, attackCatalog)}</strong>
                                            <small>{formatAttackParameters(item.attack_type, item.parameters, attackCatalog)}</small>
                                        </td>
                                        <td data-label="Resultado">
                                            <strong>{individualAnalysisSummary(item)}</strong>
                                            <small>{individualAnalysisOutcome(item)}</small>
                                        </td>
                                        <td data-label="Guardado" title={formatDateTime(item.saved_at)}>
                                            {formatCreatedAgo(item.saved_at)}
                                        </td>
                                        <td data-label="Estado">
                                            <span className="history-status preserved">Conservado</span>
                                        </td>
                                        <td className="history-open-cell profile-history-actions-cell">
                                            <button
                                                type="button"
                                                className="secondary-action small-inline"
                                                disabled={openingResultId === item.analysis_id}
                                                onClick={() => handleOpenIndividualAnalysis(item)}
                                            >
                                                {openingResultId === item.analysis_id ? 'Abriendo…' : 'Abrir'}
                                            </button>
                                            <HistoryMoreMenu
                                                label={`Más acciones para ${item.image_filename}`}
                                                menuId={`individual-${item.analysis_id}`}
                                                open={historyOpenMenuId === `individual-${item.analysis_id}`}
                                                onToggle={toggleHistoryMoreMenu}
                                            >
                                                <button
                                                    type="button"
                                                    className="danger"
                                                    disabled={historyMutationId === item.analysis_id}
                                                    onClick={() => {
                                                        setHistoryOpenMenuId(null)
                                                        setHistoryConfirmation({
                                                            type: 'delete-individual',
                                                            item,
                                                        })
                                                    }}
                                                >
                                                    <TrashIcon />
                                                    <span>Eliminar análisis</span>
                                                </button>
                                            </HistoryMoreMenu>
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

            <HistoryConfirmationDialog
                confirmation={historyConfirmation}
                busy={Boolean(historyMutationId)}
                onCancel={() => {
                    if (!historyMutationId) setHistoryConfirmation(null)
                }}
                onConfirm={handleConfirmHistoryMutation}
            />

            <section className="profile-danger-section">
                <div>
                    <span className="profile-danger-label">Zona peligrosa</span>
                    <h2>Eliminar cuenta</h2>
                    <p>
                        Elimina permanentemente tu cuenta, conjuntos de datos, evaluaciones y artefactos asociados.
                    </p>
                </div>
                <button
                    type="button"
                    className="danger-action profile-delete-trigger"
                    onClick={openDeleteDialog}
                >
                    Eliminar cuenta…
                </button>
            </section>

            {deleteDialogOpen && (
                <div className="profile-delete-backdrop">
                    <button
                        type="button"
                        className="profile-delete-backdrop-dismiss"
                        onClick={closeDeleteDialog}
                        aria-label="Cerrar diálogo de eliminación de cuenta"
                        disabled={deletingAccount}
                    />
                    <section
                        className="profile-delete-dialog"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="profile-delete-title"
                    >
                        <div className="profile-delete-icon">
                            <DangerIcon />
                        </div>
                        <div className="profile-delete-copy">
                            <span className="profile-danger-label">Acción irreversible</span>
                            <h2 id="profile-delete-title">Eliminar cuenta</h2>
                            <p>
                                Se eliminarán permanentemente tu cuenta, conjuntos de datos, evaluaciones y artefactos asociados. Confirma tu contraseña para continuar.
                            </p>
                        </div>

                        <form className="profile-delete-form" onSubmit={handleDeleteAccount}>
                            <PasswordField
                                id="delete-account-password"
                                label="Contraseña"
                                value={deletePassword}
                                onChange={(event) => {
                                    setDeletePassword(event.target.value)
                                    setDeleteError('')
                                }}
                                autoComplete="current-password"
                                placeholder="Introduce tu contraseña"
                                error={deleteError}
                                disabled={deletingAccount}
                                visible={deletePasswordVisible}
                                onVisibleChange={setDeletePasswordVisible}
                            />
                            <div className="profile-delete-actions">
                                <button
                                    type="button"
                                    className="secondary-action"
                                    onClick={closeDeleteDialog}
                                    disabled={deletingAccount}
                                >
                                    Cancelar
                                </button>
                                <button
                                    type="submit"
                                    className="danger-action"
                                    disabled={deletingAccount || !deletePassword.trim()}
                                >
                                    {deletingAccount ? 'Eliminando…' : 'Eliminar definitivamente'}
                                </button>
                            </div>
                        </form>
                    </section>
                </div>
            )}
        </section>
    )
}

export default ProfilePage
