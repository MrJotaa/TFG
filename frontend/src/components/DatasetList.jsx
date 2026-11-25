function DatasetList({
    datasets,
    selectedDataset,
    onSelectDataset,
    compact = false,
    variant = 'default',
    selectionLocked = false,
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
                        const baseClassName = variant === 'workspace'
                            ? 'dataset-workspace-option'
                            : 'sidebar-list-item'
                        const buttonClassName = `${baseClassName}${isSelected ? ' active' : ''}`
                        return (
                            <button
                                key={dataset.id}
                                type="button"
                                className={buttonClassName}
                                disabled={selectionLocked}
                                onClick={() => {
                                    if (!selectionLocked) onSelectDataset(dataset)
                                }}
                                title={selectionLocked ? 'El dataset de A permanece fijo durante la comparación' : undefined}
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
