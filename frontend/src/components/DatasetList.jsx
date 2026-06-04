function DatasetList({ datasets, selectedDataset, onSelectDataset, compact = false }) {
    return (
        <div className={compact ? 'dataset-list compact' : 'dataset-list'}>
            {!compact && <h2>Datasets</h2>}

            {datasets.length === 0 ? (
                <p>No hay datasets disponibles.</p>
            ) : (
                <div className="sidebar-list">
                    {datasets.map((dataset) => (
                        <button
                            key={dataset.id}
                            type="button"
                            className={
                                selectedDataset?.id === dataset.id
                                    ? 'sidebar-list-item active'
                                    : 'sidebar-list-item'
                            }
                            onClick={() => onSelectDataset(dataset)}
                        >
                            <span>{dataset.name}</span>
                            <small>{dataset.image_count} imágenes</small>
                        </button>
                    ))}
                </div>
            )}
        </div>
    )
}

export default DatasetList