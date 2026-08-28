function DatasetList({
    datasets,
    selectedDataset,
    onSelectDataset,
    compact = false,
    variant = 'default',
}) {
    const classNames = [
        'dataset-list',
        compact ? 'compact' : '',
        variant === 'workspace' ? 'workspace-dataset-list' : '',
    ].filter(Boolean).join(' ')

    return (
        <div className={classNames}>
            {!compact && variant !== 'workspace' && <h2>Conjuntos de datos</h2>}

            {datasets.length === 0 ? (
                <p className="dashboard-empty-copy">No hay conjuntos de datos disponibles.</p>
            ) : (
                <div
                    className={variant === 'workspace' ? 'dataset-workspace-options' : 'sidebar-list'}
                    aria-label="Conjuntos de datos disponibles"
                >
                    {datasets.map((dataset) => {
                        const isSelected = selectedDataset?.id === dataset.id
                        return (
                            <button
                                key={dataset.id}
                                type="button"
                                className={
                                    variant === 'workspace'
                                        ? `dataset-workspace-option${isSelected ? ' active' : ''}`
                                        : `sidebar-list-item${isSelected ? ' active' : ''}`
                                }
                                onClick={() => onSelectDataset(dataset)}
                                aria-pressed={isSelected}
                            >
                                <span>{dataset.name}</span>
                                <small>{dataset.image_count} imágenes</small>
                            </button>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

export default DatasetList
