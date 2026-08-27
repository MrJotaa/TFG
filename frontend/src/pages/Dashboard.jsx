import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import AttackSelector from '../components/AttackSelector'
import DashboardHistory from '../components/DashboardHistory'
import DatasetList from '../components/DatasetList'
import ImageGallery from '../components/ImageGallery'
import ModelSelector from '../components/ModelSelector'
import { getDatasetsByUser } from '../api/datasets'
import { useExperiment } from '../context/ExperimentContext'
import { useAuth } from '../context/AuthContext'
import { getAttackName } from '../utils/attackFormatting'
import { isAttackSupportedForTask } from '../utils/attackCatalog'
import { getExperimentReadiness } from '../utils/dashboard'

function DatabaseIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <ellipse cx="12" cy="5.5" rx="7" ry="3" />
            <path d="M5 5.5v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6M5 11.5v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
        </svg>
    )
}

function HistoryIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4.8 8.2A8 8 0 1 1 4 12" />
            <path d="M4 5v4h4M12 7.5V12l3 2" />
        </svg>
    )
}

function SettingsIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" />
            <path d="m18.3 13.5 1.7 1-.9 2.2-1.9-.5a7.7 7.7 0 0 1-1.5 1.5l.5 1.9-2.2.9-1-1.7a7.8 7.8 0 0 1-2 0l-1 1.7-2.2-.9.5-1.9a7.7 7.7 0 0 1-1.5-1.5l-1.9.5-.9-2.2 1.7-1a7.8 7.8 0 0 1 0-2l-1.7-1 .9-2.2 1.9.5a7.7 7.7 0 0 1 1.5-1.5l-.5-1.9 2.2-.9 1 1.7a7.8 7.8 0 0 1 2 0l1-1.7 2.2.9-.5 1.9a7.7 7.7 0 0 1 1.5 1.5l1.9-.5.9 2.2-1.7 1a7.8 7.8 0 0 1 0 2Z" />
        </svg>
    )
}

function Dashboard() {
    const navigate = useNavigate()
    const location = useLocation()
    const {
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset,
        selectedImage,
        setSelectedImage,
        activeModel,
        setActiveModel,
        attackCatalog,
        attackCatalogLoading,
        attackCatalogError,
        selectedAttack,
        setSelectedAttack,
        attackParameters,
        setAttackParameters,
        recoveredEvaluation,
        setRecoveredEvaluation,
        recoveredIndividualAnalysis,
        setRecoveredIndividualAnalysis,
        openRecoveredEvaluation,
        openRecoveredIndividualAnalysis,
    } = useExperiment()
    const { user } = useAuth()

    const [datasetsError, setDatasetsError] = useState('')
    const [imagesRefreshKey] = useState(0)
    const [leftWidth, setLeftWidth] = useState(320)
    const [workspaceTab, setWorkspaceTab] = useState(() =>
        location.state?.workspaceTab === 'history' ? 'history' : 'data',
    )

    useEffect(() => {
        async function loadUserDatasets() {
            if (!user) {
                setDatasets([])
                setSelectedDataset(null)
                setSelectedImage(null)
                return
            }

            try {
                setDatasetsError('')
                const data = await getDatasetsByUser(user.id)
                setDatasets(data)
            } catch (err) {
                setDatasetsError(err.message)
            }
        }

        loadUserDatasets()
    }, [user, setDatasets, setSelectedDataset, setSelectedImage])

    function handleDatasetSelect(dataset) {
        setSelectedDataset(dataset)
        setSelectedImage(null)
    }

    function startLeftResize(event) {
        event.preventDefault()
        const startX = event.clientX
        const startWidth = leftWidth

        function handleMouseMove(moveEvent) {
            const nextWidth = startWidth + (moveEvent.clientX - startX)
            setLeftWidth(Math.min(Math.max(nextWidth, 275), 430))
        }

        function handleMouseUp() {
            window.removeEventListener('mousemove', handleMouseMove)
            window.removeEventListener('mouseup', handleMouseUp)
        }

        window.addEventListener('mousemove', handleMouseMove)
        window.addEventListener('mouseup', handleMouseUp)
    }

    const attackSupported = Boolean(
        activeModel
        && attackCatalog.length > 0
        && isAttackSupportedForTask(
            attackCatalog,
            selectedAttack,
            activeModel.task_type || 'classification',
        )
    )

    const readiness = useMemo(
        () => getExperimentReadiness({
            selectedDataset,
            selectedImage,
            activeModel,
            attackSupported,
        }),
        [selectedDataset, selectedImage, activeModel, attackSupported],
    )

    const activeAttackName = getAttackName(selectedAttack, attackCatalog)

    function startNewEvaluation() {
        if (recoveredEvaluation) setRecoveredEvaluation(null)
        if (recoveredIndividualAnalysis) setRecoveredIndividualAnalysis(null)
        navigate('/evaluation')
    }

    function startNewIndividualAnalysis() {
        if (recoveredEvaluation) setRecoveredEvaluation(null)
        if (recoveredIndividualAnalysis) setRecoveredIndividualAnalysis(null)
        navigate('/comparison')
    }

    return (
        <div
            className="dashboard-layout dashboard-workspace-layout"
            style={{ '--left-width': `${leftWidth}px` }}
        >
            <aside className="dashboard-sidebar dashboard-config-panel" aria-label="Configuración del experimento">
                <ModelSelector onActiveModelChange={setActiveModel} />
                <AttackSelector
                    selectedAttack={selectedAttack}
                    onSelectAttack={setSelectedAttack}
                    attackParameters={attackParameters}
                    onChangeAttackParameters={setAttackParameters}
                    activeModel={activeModel}
                    attackCatalog={attackCatalog}
                    loading={attackCatalogLoading}
                    error={attackCatalogError}
                />
            </aside>

            <div
                className="resize-handle resize-handle-left dashboard-resize-handle"
                onMouseDown={startLeftResize}
                role="separator"
                aria-label="Redimensionar configuración"
                aria-orientation="vertical"
            />

            <section className="dashboard-workspace-shell">
                <nav className="dashboard-workspace-tabs" aria-label="Contenido del panel principal">
                    <button
                        type="button"
                        className={workspaceTab === 'data' ? 'active' : ''}
                        onClick={() => setWorkspaceTab('data')}
                    >
                        <DatabaseIcon />
                        <span>Datos</span>
                    </button>
                    <button
                        type="button"
                        className={workspaceTab === 'history' ? 'active' : ''}
                        onClick={() => setWorkspaceTab('history')}
                    >
                        <HistoryIcon />
                        <span>Historial</span>
                    </button>
                </nav>

                <div className="dashboard-workspace-content">
                    {workspaceTab === 'data' ? (
                        <section className="dashboard-data-view">
                            <header className="dashboard-workspace-header">
                                <div>
                                    <span className="dashboard-section-eyebrow">Espacio de trabajo</span>
                                    <h1>Datos del experimento</h1>
                                    <p>Selecciona un dataset y una imagen. La configuración activa permanece visible a la izquierda.</p>
                                </div>
                                <button
                                    type="button"
                                    className="dashboard-text-action"
                                    onClick={() => navigate('/datasets')}
                                >
                                    <SettingsIcon />
                                    <span>Gestionar datasets</span>
                                </button>
                            </header>

                            {datasetsError && <p className="error dashboard-datasets-error">{datasetsError}</p>}

                            <section className="dashboard-dataset-browser">
                                <div className="dashboard-inline-heading">
                                    <div>
                                        <h2>Conjuntos de datos</h2>
                                        <span>{datasets.length} disponibles</span>
                                    </div>
                                </div>
                                <DatasetList
                                    datasets={datasets}
                                    selectedDataset={selectedDataset}
                                    onSelectDataset={handleDatasetSelect}
                                    variant="workspace"
                                />
                            </section>

                            <section className="dashboard-image-browser">
                                <div className="dashboard-inline-heading dashboard-image-heading">
                                    <div>
                                        <h2>{selectedDataset?.name || 'Imágenes del dataset'}</h2>
                                        <span>
                                            {selectedDataset
                                                ? `${selectedDataset.image_count} imágenes · selecciona una para analizarla`
                                                : 'Selecciona un dataset para explorar sus imágenes'}
                                        </span>
                                    </div>
                                    {selectedImage && (
                                        <span className="dashboard-selected-image-label" title={selectedImage.filename}>
                                            Seleccionada · {selectedImage.filename}
                                        </span>
                                    )}
                                </div>

                                <ImageGallery
                                    selectedDataset={selectedDataset}
                                    selectedImage={selectedImage}
                                    onSelectImage={setSelectedImage}
                                    refreshKey={imagesRefreshKey}
                                    workspace
                                    allowPreview
                                    showLabelEditor={false}
                                />
                            </section>
                        </section>
                    ) : (
                        <DashboardHistory
                            datasets={datasets}
                            attackCatalog={attackCatalog}
                            openRecoveredEvaluation={openRecoveredEvaluation}
                            openRecoveredIndividualAnalysis={openRecoveredIndividualAnalysis}
                            onNavigateToEvaluation={() => navigate('/evaluation')}
                            onNavigateToIndividualAnalysis={() => navigate('/comparison')}
                            onNavigateToProfile={() => navigate('/profile')}
                            onNavigateToComparison={(left, right) => {
                                const search = new URLSearchParams({
                                    a: left.result_id,
                                    b: right.result_id,
                                })
                                navigate(`/evaluations/compare?${search.toString()}`)
                            }}
                        />
                    )}
                </div>

                <footer className="dashboard-action-bar">
                    <div className="dashboard-action-status">
                        <div className={`dashboard-readiness ${readiness.tone}`}>
                            <span aria-hidden="true" />
                            <strong>{readiness.message}</strong>
                        </div>
                        <div className="dashboard-action-context" aria-label="Configuración seleccionada">
                            <span>{activeModel?.name || 'Sin modelo activo'}</span>
                            <i aria-hidden="true">/</i>
                            <span>{activeAttackName || 'Sin ataque'}</span>
                            <i aria-hidden="true">/</i>
                            <span>{selectedDataset?.name || 'Sin dataset'}</span>
                            {selectedImage && (
                                <>
                                    <i aria-hidden="true">/</i>
                                    <span title={selectedImage.filename}>{selectedImage.filename}</span>
                                </>
                            )}
                        </div>
                    </div>

                    <div className="dashboard-action-buttons">
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={startNewEvaluation}
                            disabled={!readiness.canEvaluate}
                        >
                            Evaluar dataset
                        </button>
                        <button
                            type="button"
                            className="primary-action"
                            onClick={startNewIndividualAnalysis}
                            disabled={!readiness.canAnalyze}
                        >
                            Analizar imagen
                        </button>
                    </div>
                </footer>
            </section>
        </div>
    )
}

export default Dashboard
