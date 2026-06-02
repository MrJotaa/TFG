import AttackPanel from '../components/AttackPanel'
import { useExperiment } from '../context/ExperimentContext'

function ComparisonView() {
    const { selectedDataset, selectedImage } = useExperiment()

    return (
        <section className="view-panel">
            <h2>Comparación adversaria</h2>
            <p>
                Genera una imagen adversaria y compara predicción, perturbación y Grad-CAM.
            </p>

            <AttackPanel
                selectedDataset={selectedDataset}
                selectedImage={selectedImage}
            />
        </section>
    )
}

export default ComparisonView