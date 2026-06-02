import { useEffect, useState } from 'react'
import UserSelector from '../components/UserSelector'
import DatasetUploader from '../components/DatasetUploader'
import DatasetList from '../components/DatasetList'
import ImageGallery from '../components/ImageGallery'
import { getDatasetsByUser } from '../api/datasets'
import PredictionPanel from '../components/PredictionPanel'
import ModelSelector from '../components/ModelSelector'
import DatasetImageUploader from '../components/DatasetImageUploader'
import AttackSelector from '../components/AttackSelector'
import ExperimentSummary from '../components/ExperimentSummary'

function Dashboard() {
    const [selectedUser, setSelectedUser] = useState(null)
    const [datasets, setDatasets] = useState([])
    const [selectedDataset, setSelectedDataset] = useState(null)
    const [selectedImage, setSelectedImage] = useState(null)
    const [datasetsError, setDatasetsError] = useState('')
    const [imagesRefreshKey, setImagesRefreshKey] = useState(0)

    const [activeModel, setActiveModel] = useState(null)
    const [selectedAttack, setSelectedAttack] = useState('fgsm')
    const [attackParameters, setAttackParameters] = useState({
        epsilon: 0.03,
        alpha: 0.01,
        steps: 10,
    })

    const [activeView, setActiveView] = useState('dashboard')
    const [leftCollapsed, setLeftCollapsed] = useState(false)
    const [rightCollapsed, setRightCollapsed] = useState(false)

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
    }, [selectedUser])

    function handleDatasetUploaded(dataset) {
        setDatasets((currentDatasets) => [...currentDatasets, dataset])
    }

    function handleImagesUploaded(uploadedImages) {
        setImagesRefreshKey((current) => current + 1)

        setDatasets((currentDatasets) =>
            currentDatasets.map((dataset) =>
                dataset.id === selectedDataset.id
                    ? {
                        ...dataset,
                        image_count: dataset.image_count + uploadedImages.length,
                    }
                    : dataset,
            ),
        )

        setSelectedDataset((currentDataset) =>
            currentDataset
                ? {
                    ...currentDataset,
                    image_count: currentDataset.image_count + uploadedImages.length,
                }
                : currentDataset,
        )
    }

    function handleDatasetSelect(dataset) {
        setSelectedDataset(dataset)
        setSelectedImage(null)
    }

    return (
        <main className="app-shell">
            <header className="app-header">
                <h1>Adversarial AI App</h1>
            </header>

            <div
                className={[
                    'dashboard-layout',
                    leftCollapsed ? 'left-collapsed' : '',
                    rightCollapsed ? 'right-collapsed' : '',
                ].join(' ')}
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

                <section className="dashboard-main-panel">
                    {activeView === 'dashboard' && (
                        <ExperimentSummary
                            selectedDataset={selectedDataset}
                            selectedImage={selectedImage}
                            activeModel={activeModel}
                            selectedAttack={selectedAttack}
                            attackParameters={attackParameters}
                            onGoToPrediction={() => setActiveView('prediction')}
                        />
                    )}

                    {activeView === 'prediction' && (
                        <div className="view-panel">
                            <button
                                type="button"
                                className="secondary-action"
                                onClick={() => setActiveView('dashboard')}
                            >
                                Volver al dashboard
                            </button>

                            <PredictionPanel
                                selectedDataset={selectedDataset}
                                selectedImage={selectedImage}
                            />
                        </div>
                    )}

                    {activeView === 'datasets' && (
                        <div className="view-panel">
                            <button
                                type="button"
                                className="secondary-action"
                                onClick={() => setActiveView('dashboard')}
                            >
                                Volver al dashboard
                            </button>

                            <h2>Gestión de datasets</h2>
                            <p>
                                Vista de gestión de datasets pendiente de pulir en el
                                siguiente mockup.
                            </p>

                            <DatasetUploader
                                selectedUser={selectedUser}
                                onDatasetUploaded={handleDatasetUploaded}
                            />

                            <DatasetImageUploader
                                selectedDataset={selectedDataset}
                                onImagesUploaded={handleImagesUploaded}
                            />
                        </div>
                    )}
                </section>

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
                            <div className="user-panel">
                                <UserSelector
                                    selectedUser={selectedUser}
                                    onUserSelect={setSelectedUser}
                                />
                            </div>

                            <div className="sidebar-block">
                                <div className="sidebar-title-row">
                                    <div className="sidebar-block-title">Datasets</div>

                                    <button
                                        type="button"
                                        className="icon-button"
                                        onClick={() => setActiveView('datasets')}
                                        title="Gestionar datasets"
                                    >
                                        ⚙
                                    </button>
                                </div>

                                {datasetsError && (
                                    <p className="error">{datasetsError}</p>
                                )}

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
                                        Dataset Images
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

            <footer className="app-footer">
                © 2026 Adversarial AI App. All rights reserved.
            </footer>
        </main>
    )
}

export default Dashboard