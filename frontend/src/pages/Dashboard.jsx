import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DatasetList from '../components/DatasetList'
import ImageGallery from '../components/ImageGallery'
import { getDatasetsByUser } from '../api/datasets'
import ModelSelector from '../components/ModelSelector'
import AttackSelector from '../components/AttackSelector'
import ExperimentSummary from '../components/ExperimentSummary'
import { useExperiment } from '../context/ExperimentContext'

function Dashboard() {
    const navigate = useNavigate()

    const {
        selectedUser,
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset,
        selectedImage,
        setSelectedImage,
        activeModel,
        setActiveModel,
        selectedAttack,
        setSelectedAttack,
        attackParameters,
        setAttackParameters,
    } = useExperiment()

    const [datasetsError, setDatasetsError] = useState('')
    const [imagesRefreshKey] = useState(0)
    const [leftCollapsed, setLeftCollapsed] = useState(false)
    const [rightCollapsed, setRightCollapsed] = useState(false)
    const [leftWidth, setLeftWidth] = useState(300)
    const [rightWidth, setRightWidth] = useState(380)

    useEffect(() => {
        async function loadUserDatasets() {
            if (!selectedUser) {
                setDatasets([])
                setSelectedDataset(null)
                setSelectedImage(null)
                return
            }

            try {
                setDatasetsError('')
                const data = await getDatasetsByUser(selectedUser.id)
                setDatasets(data)
            } catch (err) {
                setDatasetsError(err.message)
            }
        }

        loadUserDatasets()
    }, [selectedUser, setDatasets, setSelectedDataset, setSelectedImage])

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
            setLeftWidth(Math.min(Math.max(nextWidth, 240), 460))
        }

        function handleMouseUp() {
            window.removeEventListener('mousemove', handleMouseMove)
            window.removeEventListener('mouseup', handleMouseUp)
        }

        window.addEventListener('mousemove', handleMouseMove)
        window.addEventListener('mouseup', handleMouseUp)
    }

    function startRightResize(event) {
        event.preventDefault()

        const startX = event.clientX
        const startWidth = rightWidth

        function handleMouseMove(moveEvent) {
            const nextWidth = startWidth - (moveEvent.clientX - startX)
            setRightWidth(Math.min(Math.max(nextWidth, 280), 520))
        }

        function handleMouseUp() {
            window.removeEventListener('mousemove', handleMouseMove)
            window.removeEventListener('mouseup', handleMouseUp)
        }

        window.addEventListener('mousemove', handleMouseMove)
        window.addEventListener('mouseup', handleMouseUp)
    }

    return (
        <div
            className={[
                'dashboard-layout',
                leftCollapsed ? 'left-collapsed' : '',
                rightCollapsed ? 'right-collapsed' : '',
            ].join(' ')}
            style={{
                '--left-width': `${leftWidth}px`,
                '--right-width': `${rightWidth}px`,
            }}
        >
            <aside className="dashboard-sidebar left-sidebar">
                <button
                    type="button"
                    className="collapse-button"
                    onClick={() => setLeftCollapsed((current) => !current)}
                    aria-label="Colapsar panel izquierdo"
                >
                    {leftCollapsed ? '>' : '<'}
                </button>

                {!leftCollapsed && (
                    <>
                        <ModelSelector onActiveModelChange={setActiveModel} />

                        <AttackSelector
                            selectedAttack={selectedAttack}
                            onSelectAttack={setSelectedAttack}
                            attackParameters={attackParameters}
                            onChangeAttackParameters={setAttackParameters}
                        />
                    </>
                )}
            </aside>

            {!leftCollapsed && (
                <div
                    className="resize-handle resize-handle-left"
                    onMouseDown={startLeftResize}
                    role="separator"
                    aria-label="Redimensionar panel izquierdo"
                />
            )}

            <section className="dashboard-main-panel">
                <ExperimentSummary
                    selectedDataset={selectedDataset}
                    selectedImage={selectedImage}
                    activeModel={activeModel}
                    selectedAttack={selectedAttack}
                    attackParameters={attackParameters}
                    onGoToPrediction={() => navigate('/prediction')}
                    onGoToEvaluation={() => navigate('/evaluation')}
                />
            </section>

            {!rightCollapsed && (
                <div
                    className="resize-handle resize-handle-right"
                    onMouseDown={startRightResize}
                    role="separator"
                    aria-label="Redimensionar panel derecho"
                />
            )}

            <aside className="dashboard-sidebar right-sidebar">
                <button
                    type="button"
                    className="collapse-button"
                    onClick={() => setRightCollapsed((current) => !current)}
                    aria-label="Colapsar panel derecho"
                >
                    {rightCollapsed ? '<' : '>'}
                </button>

                {!rightCollapsed && (
                    <>
                        <div className="dashboard-user-card">
                            <div className="user-avatar">
                                {selectedUser?.name?.charAt(0).toUpperCase() || 'U'}
                            </div>

                            <div>
                                <p className="user-card-label">Usuario activo</p>
                                <strong>{selectedUser?.name || 'Sin usuario'}</strong>
                                {selectedUser?.email && <span>{selectedUser.email}</span>}
                            </div>
                        </div>

                        <div className="sidebar-block">
                            <div className="sidebar-title-row">
                                <div className="sidebar-block-title">Datasets</div>

                                <button
                                    type="button"
                                    className="icon-button"
                                    onClick={() => navigate('/datasets')}
                                    title="Gestionar datasets"
                                >
                                    ⚙
                                </button>
                            </div>

                            {datasetsError && <p className="error">{datasetsError}</p>}

                            <DatasetList
                                datasets={datasets}
                                selectedDataset={selectedDataset}
                                onSelectDataset={handleDatasetSelect}
                                compact
                            />
                        </div>

                        <div className="sidebar-block">
                            <div className="sidebar-title-row">
                                <div className="sidebar-block-title">
                                    Imágenes del Dataset
                                </div>

                                <button
                                    type="button"
                                    className="icon-button"
                                    title="Vista ampliada pendiente"
                                >
                                    🔍
                                </button>
                            </div>

                            <ImageGallery
                                selectedDataset={selectedDataset}
                                selectedImage={selectedImage}
                                onSelectImage={setSelectedImage}
                                refreshKey={imagesRefreshKey}
                                compact
                                showLabelEditor={false}
                            />
                        </div>
                    </>
                )}
            </aside>
        </div>
    )
}

export default Dashboard