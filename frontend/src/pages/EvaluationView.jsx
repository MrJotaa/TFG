import EvaluationPanel from '../components/EvaluationPanel'
import { useExperiment } from '../context/ExperimentContext'

const ATTACK_NAMES = {
    fgsm: 'FGSM',
    random_noise: 'Random Noise',
    pgd: 'PGD',
}

function EvaluationView() {
    const {
        selectedDataset,
        activeModel,
        selectedAttack,
        attackParameters,
    } = useExperiment()

    const attackName = ATTACK_NAMES[selectedAttack] || selectedAttack

    return (
        <section className="evaluation-page">
            <div className="comparison-header">
                <div>
                    <h2>Dataset Evaluation and Analysis</h2>
                    <p>
                        Evalúa el impacto del ataque adversario sobre todas las imágenes
                        del dataset seleccionado.
                    </p>
                </div>
            </div>

            <div className="context-summary">
                <h3>Context Summary</h3>

                <div className="context-grid">
                    <p>
                        <strong>Dataset:</strong>{' '}
                        {selectedDataset?.name || 'No dataset selected'}
                    </p>

                    <p>
                        <strong>Total images:</strong>{' '}
                        {selectedDataset?.image_count ?? '-'}
                    </p>

                    <p>
                        <strong>Model:</strong>{' '}
                        {activeModel?.name || 'No active model'}
                    </p>

                    <p>
                        <strong>Attack:</strong> {attackName}
                    </p>

                    <p>
                        <strong>Parameters:</strong> ε={attackParameters.epsilon}
                        {selectedAttack === 'pgd' &&
                            ` · α=${attackParameters.alpha} · steps=${attackParameters.steps}`}
                    </p>
                </div>
            </div>

            <EvaluationPanel
                selectedDataset={selectedDataset}
                selectedAttack={selectedAttack}
                attackParameters={attackParameters}
            />
        </section>
    )
}

export default EvaluationView