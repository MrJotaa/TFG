import EvaluationPanel from '../components/EvaluationPanel'
import { useExperiment } from '../context/ExperimentContext'

function EvaluationView() {
    const { selectedDataset } = useExperiment()

    return (
        <section className="view-panel">
            <h2>Evaluación del dataset</h2>
            <p>
                Ejecuta ataques sobre todo el dataset y analiza métricas agregadas.
            </p>

            <EvaluationPanel selectedDataset={selectedDataset} />
        </section>
    )
}

export default EvaluationView