import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'

function ExperimentSummary({
    selectedDataset,
    selectedImage,
    activeModel,
    selectedAttack,
    attackParameters,
    attackCatalog,
    onGoToEvaluation,
    onGoToComparison,
}) {
    const selectedTarget = selectedImage
        ? selectedImage.filename
        : selectedDataset
            ? selectedDataset.name
            : 'Sin selección'

    return (
        <section className="experiment-summary">
            <div className="summary-heading">
                <h2>Resumen del Experimento</h2>
                <p>
                    Configura el experimento seleccionando conjunto de datos, imagen, modelo y
                    ataque antes de ejecutar el análisis.
                </p>
            </div>

            <div className="summary-grid">
                <article className="summary-card summary-card-blue">
                    <span>Imagen / conjunto de datos seleccionado</span>
                    <strong>{selectedTarget}</strong>
                </article>

                <article className="summary-card summary-card-green">
                    <span>Modelo activo</span>
                    <strong>{activeModel?.name || 'Sin modelo activo'}</strong>
                </article>

                <article className="summary-card summary-card-yellow">
                    <span>Ataque activo</span>
                    <strong>{getAttackName(selectedAttack, attackCatalog)}</strong>
                </article>

                <article className="summary-card summary-card-red">
                    <span>Parámetros del ataque</span>
                    <strong>
                        {formatAttackParameters(selectedAttack, attackParameters, attackCatalog)}
                    </strong>
                </article>
            </div>

            <div className="summary-actions">
                <button
                    type="button"
                    className="primary-action"
                    onClick={onGoToComparison}
                    disabled={!selectedDataset || !selectedImage || !activeModel}
                >
                    Analizar imagen
                </button>

                <button
                    type="button"
                    className="secondary-action"
                    onClick={onGoToEvaluation}
                    disabled={!selectedDataset || !activeModel}
                >
                    Evaluar
                </button>
            </div>
        </section>
    )
}

export default ExperimentSummary
