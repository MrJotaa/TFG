import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DatasetUploader from '../components/DatasetUploader'
import DatasetImageUploader from '../components/DatasetImageUploader'
import ImageGallery from '../components/ImageGallery'
import { deleteDataset, getDatasetsByUser } from '../api/datasets'
import { useExperiment } from '../context/ExperimentContext'
import { useAuth } from '../context/AuthContext'

function DatasetManager() {
    const navigate = useNavigate()

    const {
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset,
        selectedImage,
        setSelectedImage,
    } = useExperiment()

    const [datasetsError, setDatasetsError] = useState('')
    const [imagesRefreshKey, setImagesRefreshKey] = useState(0)
    const { user } = useAuth()

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

    async function handleDeleteDataset(datasetId) {
        const confirmed = window.confirm(
            '¿Seguro que quieres borrar este dataset? Esta acción no se puede deshacer.',
        )

        if (!confirmed) return

        try {
            setDatasetsError('')
            await deleteDataset(datasetId)

            setDatasets((currentDatasets) =>
                currentDatasets.filter((dataset) => dataset.id !== datasetId),
            )

            if (selectedDataset?.id === datasetId) {
                setSelectedDataset(null)
                setSelectedImage(null)
            }
        } catch (err) {
            setDatasetsError(err.message)
        }
    }

    return (
        <div className="dataset-manager-page">
            <div className="page-toolbar">
                <div>
                    <h2>Gestión de Datasets</h2>
                    <p>Gestiona datasets, imágenes y etiquetas asociadas.</p>
                </div>

                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard')}
                >
                    Volver al Dashboard
                </button>
            </div>

            {datasetsError && <p className="error">{datasetsError}</p>}

            <div className="dataset-manager-layout">
                <aside className="dataset-manager-sidebar">
                    <section className="management-card">
                        <h3>My Datasets</h3>

                        {!user ? (
                            <p>Inicia sesión para ver tus datasets.</p>
                        ) : datasets.length === 0 ? (
                            <p>No hay datasets disponibles.</p>
                        ) : (
                            <div className="management-list">
                                {datasets.map((dataset) => (
                                    <article
                                        key={dataset.id}
                                        className={
                                            selectedDataset?.id === dataset.id
                                                ? 'dataset-management-item active'
                                                : 'dataset-management-item'
                                        }
                                    >
                                        <h4>{dataset.name}</h4>
                                        <p>
                                            Imágenes:{' '}
                                            <strong>{dataset.image_count}</strong>
                                        </p>

                                        <div className="item-actions">
                                            <button
                                                type="button"
                                                className="secondary-action small"
                                                onClick={() => {
                                                    setSelectedDataset(dataset)
                                                    setSelectedImage(null)
                                                }}
                                            >
                                                Seleccionar
                                            </button>

                                            <button
                                                type="button"
                                                className="danger-action small"
                                                onClick={() =>
                                                    handleDeleteDataset(dataset.id)
                                                }
                                            >
                                                Eliminar
                                            </button>
                                        </div>
                                    </article>
                                ))}
                            </div>
                        )}
                    </section>

                    <section className="management-card">
                        <h3>Subir un Nuevo Dataset</h3>

                        <DatasetUploader
                            selectedUser={user}
                            onDatasetUploaded={handleDatasetUploaded}
                        />

                        <p className="muted-text">
                            Soporte para subida de archivos ZIP disponible pronto.
                        </p>
                    </section>
                </aside>

                <section className="dataset-manager-content">
                    <div className="management-card">
                        <h3>
                            Dataset Seleccionado:{' '}
                            <span>{selectedDataset?.name || 'None'}</span>
                        </h3>

                        {!selectedDataset ? (
                            <p>Selecciona un dataset para gestionar sus imágenes.</p>
                        ) : (
                            <>
                                <p>
                                    Imágenes Totales:{' '}
                                    <strong>{selectedDataset.image_count}</strong>
                                </p>

                                <DatasetImageUploader
                                    selectedDataset={selectedDataset}
                                    onImagesUploaded={handleImagesUploaded}
                                />

                                <ImageGallery
                                    selectedDataset={selectedDataset}
                                    selectedImage={selectedImage}
                                    onSelectImage={setSelectedImage}
                                    refreshKey={imagesRefreshKey}
                                    showLabelEditor
                                    showDeleteButton
                                    onImageDeleted={() => {
                                        setImagesRefreshKey((current) => current + 1)

                                        setDatasets((currentDatasets) =>
                                            currentDatasets.map((dataset) =>
                                                dataset.id === selectedDataset.id
                                                    ? {
                                                        ...dataset,
                                                        image_count: Math.max(
                                                            dataset.image_count - 1,
                                                            0,
                                                        ),
                                                    }
                                                    : dataset,
                                            ),
                                        )

                                        setSelectedDataset((currentDataset) =>
                                            currentDataset
                                                ? {
                                                    ...currentDataset,
                                                    image_count: Math.max(
                                                        currentDataset.image_count - 1,
                                                        0,
                                                    ),
                                                }
                                                : currentDataset,
                                        )
                                    }}
                                />
                            </>
                        )}
                    </div>
                </section>
            </div>
        </div>
    )
}

export default DatasetManager