import { useCallback, useMemo, useState, useEffect } from 'react'
import BulkImagePicker from '../components/BulkImagePicker'
import CocoAnnotationPanel from '../components/CocoAnnotationPanel'
import DatasetDialog from '../components/DatasetDialog'
import ImageGallery from '../components/ImageGallery'
import {
    addImagesToDataset,
    deleteDataset,
    deleteDatasetImage,
    getDatasetsByUser,
    updateDataset,
    updateImageLabel,
    uploadDataset,
} from '../api/datasets'
import { useExperiment } from '../context/ExperimentContext'
import { useAuth } from '../context/AuthContext'
import {
    datasetSupportsAnnotations,
    getClassificationLabelStats,
    getDatasetSidebarMeta,
    getDatasetTaskLabel,
} from '../utils/datasetManagement'
import { getAnnotationStatusLabel } from '../utils/cocoAnnotations'

function PlusIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 5v14M5 12h14" />
        </svg>
    )
}

function EditIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m4.5 19.5 3.7-.8L18.4 8.5l-2.9-2.9L5.3 15.8l-.8 3.7ZM13.9 7.2l2.9 2.9" />
        </svg>
    )
}

function TrashIcon() {
    return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 7h14M9 7V4h6v3m2 0-.7 13H7.7L7 7m3.2 3v6.5m3.6-6.5v6.5" />
        </svg>
    )
}

function imageSource(image) {
    return `http://127.0.0.1:8000${image.url}`
}

function DatasetCreateDialog({ user, onClose, onCreated }) {
    const [name, setName] = useState('')
    const [taskType, setTaskType] = useState('classification')
    const [files, setFiles] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function handleSubmit(event) {
        event.preventDefault()
        if (!name.trim() || files.length === 0 || !user) return

        try {
            setLoading(true)
            setError('')
            const dataset = await uploadDataset({
                userId: user.id,
                name: name.trim(),
                taskType,
                files,
            })
            onCreated(dataset)
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setLoading(false)
        }
    }

    const imageNoun = files.length === 1 ? 'imagen' : 'imágenes'
    const createLabel = loading ? 'Creando conjunto...' : `Crear con ${files.length} ${imageNoun}`

    return (
        <DatasetDialog
            title="Nuevo conjunto de datos"
            eyebrow="Creación"
            onClose={onClose}
            busy={loading}
        >
            <form className="dataset-dialog-form" onSubmit={handleSubmit}>
                <label className="dataset-form-field">
                    <span>Nombre</span>
                    <input
                        type="text"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        placeholder="Ej. Imagenette 100 balanced"
                        disabled={loading}
                        autoFocus
                    />
                </label>

                <fieldset className="dataset-task-picker">
                    <legend>Tipo de conjunto</legend>
                    <label className={taskType === 'classification' ? 'active' : ''}>
                        <input
                            type="radio"
                            name="dataset-task"
                            value="classification"
                            checked={taskType === 'classification'}
                            onChange={() => setTaskType('classification')}
                            disabled={loading}
                        />
                        <strong>Clasificación</strong>
                        <span>Una etiqueta de clase por imagen.</span>
                    </label>
                    <label className={taskType === 'detection' ? 'active' : ''}>
                        <input
                            type="radio"
                            name="dataset-task"
                            value="detection"
                            checked={taskType === 'detection'}
                            onChange={() => setTaskType('detection')}
                            disabled={loading}
                        />
                        <strong>Detección</strong>
                        <span>Anotaciones de objetos mediante COCO.</span>
                    </label>
                </fieldset>

                <div className="dataset-form-field">
                    <span>Imágenes</span>
                    <BulkImagePicker files={files} onChange={setFiles} disabled={loading} />
                </div>

                {error && <p className="error dataset-inline-feedback">{error}</p>}

                <div className="dataset-dialog-actions">
                    <button type="button" className="secondary-action" onClick={onClose} disabled={loading}>
                        Cancelar
                    </button>
                    <button
                        type="submit"
                        className="primary-action"
                        disabled={loading || !name.trim() || files.length === 0}
                    >
                        {createLabel}
                    </button>
                </div>
            </form>
        </DatasetDialog>
    )
}

function AddImagesDialog({ dataset, onClose, onUploaded }) {
    const [files, setFiles] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function handleSubmit(event) {
        event.preventDefault()
        if (files.length === 0) return

        try {
            setLoading(true)
            setError('')
            const images = await addImagesToDataset({ datasetId: dataset.id, files })
            onUploaded(images)
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setLoading(false)
        }
    }

    const imageNoun = files.length === 1 ? 'imagen' : 'imágenes'
    const addLabel = loading ? 'Añadiendo...' : `Añadir ${files.length || ''} ${imageNoun}`

    return (
        <DatasetDialog
            title={`Añadir imágenes a ${dataset.name}`}
            eyebrow="Subida masiva"
            onClose={onClose}
            busy={loading}
        >
            <form className="dataset-dialog-form" onSubmit={handleSubmit}>
                <BulkImagePicker files={files} onChange={setFiles} disabled={loading} />
                <p className="dataset-dialog-help">
                    Puedes seleccionar muchas imágenes en una única operación. Todas se añadirán al conjunto actual.
                </p>
                {error && <p className="error dataset-inline-feedback">{error}</p>}
                <div className="dataset-dialog-actions">
                    <button type="button" className="secondary-action" onClick={onClose} disabled={loading}>
                        Cancelar
                    </button>
                    <button type="submit" className="primary-action" disabled={loading || files.length === 0}>
                        {addLabel}
                    </button>
                </div>
            </form>
        </DatasetDialog>
    )
}

function EditDatasetDialog({ dataset, onClose, onUpdated }) {
    const [name, setName] = useState(dataset.name)
    const [taskType, setTaskType] = useState(dataset.task_type || '')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function handleSubmit(event) {
        event.preventDefault()
        if (!name.trim()) return

        try {
            setLoading(true)
            setError('')
            const updated = await updateDataset({
                datasetId: dataset.id,
                name: name.trim(),
                taskType: dataset.task_type ? undefined : (taskType || undefined),
            })
            onUpdated(updated)
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <DatasetDialog title="Editar conjunto de datos" eyebrow="Metadatos" onClose={onClose} busy={loading}>
            <form className="dataset-dialog-form" onSubmit={handleSubmit}>
                <label className="dataset-form-field">
                    <span>Nombre</span>
                    <input value={name} onChange={(event) => setName(event.target.value)} disabled={loading} autoFocus />
                </label>

                {dataset.task_type ? (
                    <div className="dataset-readonly-type">
                        <span>Tipo</span>
                        <strong>{getDatasetTaskLabel(dataset.task_type)}</strong>
                        <small>El tipo queda fijado una vez definido para evitar mezclar anotaciones incompatibles.</small>
                    </div>
                ) : (
                    <label className="dataset-form-field">
                        <span>Tipo</span>
                        <select value={taskType} onChange={(event) => setTaskType(event.target.value)} disabled={loading}>
                            <option value="">Sin definir</option>
                            <option value="classification">Clasificación</option>
                            <option value="detection">Detección</option>
                        </select>
                    </label>
                )}

                {error && <p className="error dataset-inline-feedback">{error}</p>}
                <div className="dataset-dialog-actions">
                    <button type="button" className="secondary-action" onClick={onClose} disabled={loading}>Cancelar</button>
                    <button type="submit" className="primary-action" disabled={loading || !name.trim()}>
                        {loading ? 'Guardando...' : 'Guardar cambios'}
                    </button>
                </div>
            </form>
        </DatasetDialog>
    )
}

function ConfirmDeleteDialog({ title, description, confirmLabel, onClose, onConfirm, busy }) {
    return (
        <DatasetDialog title={title} eyebrow="Acción destructiva" onClose={onClose} busy={busy} danger>
            <div className="dataset-dialog-body"><p>{description}</p></div>
            <div className="dataset-dialog-actions">
                <button type="button" className="secondary-action" onClick={onClose} disabled={busy}>Cancelar</button>
                <button type="button" className="danger-action" onClick={onConfirm} disabled={busy}>
                    {busy ? 'Eliminando...' : confirmLabel}
                </button>
            </div>
        </DatasetDialog>
    )
}

function DatasetImageInspector({ dataset, image, onSaved, onDelete }) {
    const [label, setLabel] = useState(image.label || '')
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState('')
    const classification = dataset.task_type === 'classification'
    let taskTypeLabel = 'Sin definir'
    if (classification) taskTypeLabel = 'Clasificación'
    else if (dataset.task_type === 'detection') taskTypeLabel = 'Detección'

    async function handleSave(event) {
        event.preventDefault()
        if (!classification) return

        try {
            setSaving(true)
            setError('')
            const updated = await updateImageLabel({
                datasetId: dataset.id,
                imageId: image.id,
                label,
            })
            onSaved(updated)
        } catch (requestError) {
            setError(requestError.message)
        } finally {
            setSaving(false)
        }
    }

    return (
        <aside className="dataset-image-inspector">
            <div className="dataset-inspector-heading">
                <span className="dashboard-section-eyebrow">Detalle de imagen</span>
                <strong title={image.filename}>{image.filename}</strong>
            </div>
            <img className="dataset-inspector-preview" src={imageSource(image)} alt={image.filename} />
            <dl className="dataset-image-metadata">
                <div><dt>Dimensiones</dt><dd>{image.width && image.height ? `${image.width} × ${image.height}` : 'No disponibles'}</dd></div>
                <div><dt>Tipo</dt><dd>{taskTypeLabel}</dd></div>
            </dl>

            {classification && (
                <form className="dataset-label-editor-modern" onSubmit={handleSave}>
                    <label>
                        <span>Etiqueta de clase</span>
                        <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Ej. tench" disabled={saving} />
                    </label>
                    {error && <p className="error dataset-inline-feedback">{error}</p>}
                    <button type="submit" className="primary-action" disabled={saving || label.trim() === (image.label || '').trim()}>
                        {saving ? 'Guardando...' : 'Guardar etiqueta'}
                    </button>
                </form>
            )}

            {dataset.task_type === 'detection' && (
                <div className="dataset-detection-image-status">
                    <span>Estado de anotación</span>
                    <strong>{getAnnotationStatusLabel(image.annotation_review_status)}</strong>
                    <small>Las cajas y categorías se gestionan mediante el snapshot COCO del conjunto.</small>
                </div>
            )}

            {!dataset.task_type && (
                <p className="dataset-inspector-note">Define el tipo de conjunto para habilitar sus herramientas de etiquetado.</p>
            )}

            <button type="button" className="dataset-delete-image-action" onClick={() => onDelete(image)}>
                <TrashIcon /> Eliminar imagen
            </button>
        </aside>
    )
}

function DatasetLibrarySidebar({ datasets, selectedDataset, onSelect, onCreate }) {
    return (
        <aside className="dataset-library-sidebar">
            <div className="dataset-library-heading">
                <div><span>CONJUNTOS</span><small>{datasets.length} disponibles</small></div>
                <button type="button" className="dataset-new-button" onClick={onCreate}><PlusIcon /> Nuevo</button>
            </div>
            {datasets.length === 0 ? (
                <div className="dataset-library-empty"><strong>Todavía no hay conjuntos</strong><span>Crea el primero para empezar a trabajar con imágenes.</span></div>
            ) : (
                <nav className="dataset-library-list" aria-label="Conjuntos de datos">
                    {datasets.map((dataset) => (
                        <button type="button" key={dataset.id} className={selectedDataset?.id === dataset.id ? 'active' : ''} onClick={() => onSelect(dataset)}>
                            <strong>{dataset.name}</strong><span>{getDatasetSidebarMeta(dataset)}</span>
                        </button>
                    ))}
                </nav>
            )}
        </aside>
    )
}

function DatasetImagesWorkspace({ selectedDataset, classificationStats, imageFilter, setImageFilter, selectedImage, setSelectedImage, imagesRefreshKey, handleImagesLoaded, filterImage, handleImageLabelSaved, onDeleteImage }) {
    let galleryDescription = 'Selecciona una imagen para inspeccionarla.'
    if (selectedDataset.task_type === 'classification') {
        galleryDescription = `${classificationStats.labeled} de ${classificationStats.total} etiquetadas · ${classificationStats.percentage}% de cobertura`
    } else if (selectedDataset.task_type === 'detection') {
        galleryDescription = 'Selecciona una imagen para consultar su estado dentro del snapshot COCO.'
    }

    return (
        <div className="dataset-images-workspace">
            <section className="dataset-gallery-panel">
                <div className="dataset-gallery-heading">
                    <div><h3>Imágenes</h3><p>{galleryDescription}</p></div>
                    {selectedDataset.task_type === 'classification' && (
                        <div className="dataset-image-filters" aria-label="Filtrar imágenes">
                            {[
                                ['all', `Todas (${classificationStats.total})`],
                                ['labeled', `Etiquetadas (${classificationStats.labeled})`],
                                ['unlabeled', `Sin etiquetar (${classificationStats.unlabeled})`],
                            ].map(([value, label]) => (
                                <button key={value} type="button" className={imageFilter === value ? 'active' : ''} onClick={() => setImageFilter(value)}>{label}</button>
                            ))}
                        </div>
                    )}
                </div>
                <ImageGallery selectedDataset={selectedDataset} selectedImage={selectedImage} onSelectImage={setSelectedImage} refreshKey={imagesRefreshKey} workspace allowPreview onImagesLoaded={handleImagesLoaded} filterImage={filterImage} />
            </section>
            {selectedImage && <DatasetImageInspector key={`${selectedImage.id}:${selectedImage.label || ''}:${selectedImage.annotation_review_status || ''}`} dataset={selectedDataset} image={selectedImage} onSaved={handleImageLabelSaved} onDelete={onDeleteImage} />}
        </div>
    )
}

function SelectedDatasetWorkspace({ selectedDataset, workspaceTab, setWorkspaceTab, classificationStats, imageFilter, setImageFilter, selectedImage, setSelectedImage, imagesRefreshKey, handleImagesLoaded, filterImage, handleImageLabelSaved, onAssignType, onOpenDialog }) {
    const annotationsEnabled = datasetSupportsAnnotations(selectedDataset)
    return (
        <>
            <header className="dataset-selected-heading">
                <div><div className="dataset-title-line"><h2>{selectedDataset.name}</h2><span className={`dataset-status-badge ${selectedDataset.task_type || 'undefined'}`}>{getDatasetTaskLabel(selectedDataset.task_type)}</span></div><p>{selectedDataset.image_count} {selectedDataset.image_count === 1 ? 'imagen' : 'imágenes'}</p></div>
                <div className="dataset-heading-actions">
                    <button type="button" className="secondary-action dataset-icon-action" onClick={() => onOpenDialog('edit')}><EditIcon /> Editar</button>
                    <button type="button" className="primary-action dataset-add-images" onClick={() => onOpenDialog('add-images')}><PlusIcon /> Añadir imágenes</button>
                    <button type="button" className="dataset-more-action" onClick={() => onOpenDialog('delete-dataset')} aria-label="Eliminar conjunto de datos" title="Eliminar conjunto de datos"><TrashIcon /></button>
                </div>
            </header>
            {!selectedDataset.task_type && (
                <section className="dataset-type-resolution">
                    <div><span className="dashboard-section-eyebrow">Conjunto histórico</span><h3>Define cómo se utiliza este conjunto</h3><p>Fue creado antes de introducir los tipos de dataset. Esta decisión habilita únicamente las herramientas compatibles.</p></div>
                    <div><button type="button" className="secondary-action" onClick={() => onAssignType('classification')}>Clasificación</button><button type="button" className="secondary-action" onClick={() => onAssignType('detection')}>Detección</button></div>
                </section>
            )}
            <div className="dataset-workspace-tabs" role="tablist" aria-label="Gestión del conjunto">
                <button type="button" className={workspaceTab === 'images' ? 'active' : ''} onClick={() => setWorkspaceTab('images')}>Imágenes</button>
                {annotationsEnabled && <button type="button" className={workspaceTab === 'annotations' ? 'active' : ''} onClick={() => setWorkspaceTab('annotations')}>Anotaciones</button>}
            </div>
            {workspaceTab === 'annotations' && annotationsEnabled ? (
                <div className="dataset-annotations-workspace"><CocoAnnotationPanel selectedDataset={selectedDataset} onAnnotationsImported={() => onOpenDialog('refresh-images')} /></div>
            ) : (
                <DatasetImagesWorkspace selectedDataset={selectedDataset} classificationStats={classificationStats} imageFilter={imageFilter} setImageFilter={setImageFilter} selectedImage={selectedImage} setSelectedImage={setSelectedImage} imagesRefreshKey={imagesRefreshKey} handleImagesLoaded={handleImagesLoaded} filterImage={filterImage} handleImageLabelSaved={handleImageLabelSaved} onDeleteImage={(image) => { setSelectedImage(image); onOpenDialog('delete-image') }} />
            )}
        </>
    )
}

function DatasetWorkspaceMain(props) {
    if (!props.selectedDataset) {
        return <div className="dataset-workspace-empty"><span className="dataset-workspace-empty-icon"><PlusIcon /></span><strong>Selecciona o crea un conjunto de datos</strong><p>Cuando elijas uno, sus imágenes y herramientas de gestión aparecerán aquí.</p><button type="button" className="primary-action" onClick={() => props.onOpenDialog('create')}>Crear conjunto</button></div>
    }
    return <SelectedDatasetWorkspace {...props} />
}

function DatasetManagerDialogs({ dialog, user, selectedDataset, selectedImage, deleting, onClose, onCreated, onUploaded, onUpdated, onDeleteDataset, onDeleteImage }) {
    if (dialog === 'create') return <DatasetCreateDialog user={user} onClose={onClose} onCreated={onCreated} />
    if (dialog === 'add-images' && selectedDataset) return <AddImagesDialog dataset={selectedDataset} onClose={onClose} onUploaded={onUploaded} />
    if (dialog === 'edit' && selectedDataset) return <EditDatasetDialog dataset={selectedDataset} onClose={onClose} onUpdated={onUpdated} />
    if (dialog === 'delete-dataset' && selectedDataset) return <ConfirmDeleteDialog title={`Eliminar ${selectedDataset.name}`} description={`Se eliminarán permanentemente sus ${selectedDataset.image_count} imágenes, etiquetas, anotaciones y evaluaciones asociadas. Esta acción no se puede deshacer.`} confirmLabel="Eliminar conjunto" onClose={onClose} onConfirm={onDeleteDataset} busy={deleting} />
    if (dialog === 'delete-image' && selectedImage) return <ConfirmDeleteDialog title="Eliminar imagen" description={`Se eliminará permanentemente ${selectedImage.filename} y sus anotaciones asociadas.`} confirmLabel="Eliminar imagen" onClose={onClose} onConfirm={onDeleteImage} busy={deleting} />
    return null
}

function DatasetManager() {
    const {
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset,
        selectedImage,
        setSelectedImage,
    } = useExperiment()
    const { user } = useAuth()
    const [datasetsError, setDatasetsError] = useState('')
    const [imagesRefreshKey, setImagesRefreshKey] = useState(0)
    const [loadedImages, setLoadedImages] = useState([])
    const [workspaceTab, setWorkspaceTab] = useState('images')
    const [imageFilter, setImageFilter] = useState('all')
    const [dialog, setDialog] = useState(null)
    const [deleting, setDeleting] = useState(false)

    useEffect(() => {
        let active = true
        if (!user) return undefined

        getDatasetsByUser(user.id)
            .then((data) => {
                if (!active) return
                setDatasetsError('')
                setDatasets(data)
                setSelectedDataset((current) => {
                    if (current) {
                        return data.find((dataset) => dataset.id === current.id) || data[0] || null
                    }
                    return data[0] || null
                })
            })
            .catch((requestError) => {
                if (active) setDatasetsError(requestError.message)
            })

        return () => { active = false }
    }, [user, setDatasets, setSelectedDataset])

    const handleImagesLoaded = useCallback((images) => {
        setLoadedImages(images)
        setSelectedImage((current) => {
            if (!current) return null
            return images.find((image) => image.id === current.id) || null
        })
    }, [setSelectedImage])

    const classificationStats = useMemo(
        () => getClassificationLabelStats(loadedImages),
        [loadedImages],
    )

    const filterImage = (image) => {
        if (selectedDataset?.task_type !== 'classification' || imageFilter === 'all') return true
        const labeled = Boolean(image.label?.trim())
        return imageFilter === 'labeled' ? labeled : !labeled
    }

    function replaceDataset(updatedDataset) {
        setDatasets((current) => current.map((dataset) => dataset.id === updatedDataset.id ? updatedDataset : dataset))
        setSelectedDataset((current) => current?.id === updatedDataset.id ? updatedDataset : current)
    }

    function selectDataset(dataset) {
        setSelectedDataset(dataset)
        setSelectedImage(null)
        setLoadedImages([])
        setWorkspaceTab('images')
        setImageFilter('all')
    }

    function handleDatasetCreated(dataset) {
        setDatasets((current) => [...current, dataset])
        selectDataset(dataset)
        setDialog(null)
    }

    function handleImagesUploaded(images) {
        const updated = {
            ...selectedDataset,
            image_count: selectedDataset.image_count + images.length,
        }
        replaceDataset(updated)
        setImagesRefreshKey((current) => current + 1)
        setDialog(null)
    }

    async function handleAssignType(taskType) {
        if (!selectedDataset) return
        try {
            setDatasetsError('')
            const updated = await updateDataset({ datasetId: selectedDataset.id, taskType })
            replaceDataset(updated)
            setWorkspaceTab('images')
        } catch (requestError) {
            setDatasetsError(requestError.message)
        }
    }

    async function handleDeleteDataset() {
        if (!selectedDataset) return
        try {
            setDeleting(true)
            setDatasetsError('')
            await deleteDataset(selectedDataset.id)
            const remaining = datasets.filter((dataset) => dataset.id !== selectedDataset.id)
            setDatasets(remaining)
            setSelectedDataset(remaining[0] || null)
            setSelectedImage(null)
            setWorkspaceTab('images')
            setImageFilter('all')
            setDialog(null)
        } catch (requestError) {
            setDatasetsError(requestError.message)
        } finally {
            setDeleting(false)
        }
    }

    async function handleDeleteImage() {
        if (!selectedDataset || !selectedImage) return
        try {
            setDeleting(true)
            setDatasetsError('')
            await deleteDatasetImage({ datasetId: selectedDataset.id, imageId: selectedImage.id })
            replaceDataset({ ...selectedDataset, image_count: Math.max(selectedDataset.image_count - 1, 0) })
            setSelectedImage(null)
            setImagesRefreshKey((current) => current + 1)
            setDialog(null)
        } catch (requestError) {
            setDatasetsError(requestError.message)
        } finally {
            setDeleting(false)
        }
    }

    function handleImageLabelSaved(updatedImage) {
        setSelectedImage((current) => current?.id === updatedImage.id ? { ...current, ...updatedImage } : current)
        setImagesRefreshKey((current) => current + 1)
    }

    function handleOpenDialog(value) {
        if (value === 'refresh-images') {
            setImagesRefreshKey((current) => current + 1)
            return
        }
        setDialog(value)
    }

    return (
        <div className="dataset-manager-page modern">
            <header className="dataset-manager-heading"><span className="dashboard-section-eyebrow">Biblioteca de datos</span><h1>Conjuntos de datos</h1><p>Organiza imágenes, etiquetas de clasificación y anotaciones de detección desde un único espacio.</p></header>
            {datasetsError && <p className="error dataset-page-error">{datasetsError}</p>}
            <div className="dataset-workspace-shell">
                <DatasetLibrarySidebar datasets={datasets} selectedDataset={selectedDataset} onSelect={selectDataset} onCreate={() => setDialog('create')} />
                <main className="dataset-management-workspace">
                    <DatasetWorkspaceMain
                        selectedDataset={selectedDataset}
                        workspaceTab={workspaceTab}
                        setWorkspaceTab={setWorkspaceTab}
                        classificationStats={classificationStats}
                        imageFilter={imageFilter}
                        setImageFilter={setImageFilter}
                        selectedImage={selectedImage}
                        setSelectedImage={setSelectedImage}
                        imagesRefreshKey={imagesRefreshKey}
                        handleImagesLoaded={handleImagesLoaded}
                        filterImage={filterImage}
                        handleImageLabelSaved={handleImageLabelSaved}
                        onAssignType={handleAssignType}
                        onOpenDialog={handleOpenDialog}
                    />
                </main>
            </div>
            <DatasetManagerDialogs
                dialog={dialog}
                user={user}
                selectedDataset={selectedDataset}
                selectedImage={selectedImage}
                deleting={deleting}
                onClose={() => setDialog(null)}
                onCreated={handleDatasetCreated}
                onUploaded={handleImagesUploaded}
                onUpdated={(updated) => { replaceDataset(updated); setDialog(null) }}
                onDeleteDataset={handleDeleteDataset}
                onDeleteImage={handleDeleteImage}
            />
        </div>
    )
}

export default DatasetManager
