function DatasetList({ datasets, selectedDataset, onSelectDataset }) {
    return (
        <div>
            <h2>Datasets</h2>

            {datasets.length === 0 ? (
                <p>No hay datasets disponibles.</p>
            ) : (
                <ul>
                    {datasets.map((dataset) => (
                        <li key={dataset.id}>
                            <button
                                onClick={() => onSelectDataset(dataset)}
                                style={{
                                    fontWeight:
                                        selectedDataset?.id === dataset.id ? 'bold' : 'normal',
                                }}
                            >
                                {dataset.name} — {dataset.image_count} imágenes
                            </button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    )
}

export default DatasetList