const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
}

function ExperimentSummary({
    selectedDataset,
    selectedImage,
    activeModel,
    selectedAttack,
    attackParameters,
    onGoToPrediction,
}) {
    const selectedTarget = selectedImage
        ? selectedImage.filename
        : selectedDataset
            ? selectedDataset.name
            : 'Sin selección'

    return (
        <section className="experiment-summary">
            <div className="summary-heading">
                <h2>Predict Overview</h2>
                <p>
                    Configura el experimento seleccionando usuario, dataset, imagen,
                    modelo y ataque antes de ejecutar la predicción.
                </p>
            </div>

            <div className="summary-grid">
                <article className="summary-card summary-card-blue">
                    <span>Imagen / dataset seleccionado</span>
                    <strong>{selectedTarget}</strong>
                </article>

                <article className="summary-card summary-card-green">
                    <span>Modelo activo</span>
                    <strong>{activeModel?.name || 'Sin modelo activo'}</strong>
                </article>

                <article className="summary-card summary-card-yellow">
                    <span>Ataque activo</span>
                    <strong>{ATTACK_NAMES[selectedAttack] || selectedAttack}</strong>
                </article>

                <article className="summary-card summary-card-red">
                    <span>Parámetros del ataque</span>
                    <strong>
                        ε={attackParameters.epsilon}
                        {selectedAttack === 'pgd' &&
                            ` · α=${attackParameters.alpha} · steps=${attackParameters.steps}`}
                    </strong>
                </article>
            </div>

            <div className="summary-actions">
                <button
                    type="button"
                    className="primary-action"
                    onClick={onGoToPrediction}
                    disabled={!selectedDataset || !selectedImage || !activeModel}
                >
                    Predict
                </button>
            </div>
        </section>
    )
}

export default ExperimentSummary