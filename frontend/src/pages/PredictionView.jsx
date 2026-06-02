import PredictionPanel from '../components/PredictionPanel'
import { useExperiment } from '../context/ExperimentContext'

function PredictionView() {
    const { selectedDataset, selectedImage } = useExperiment()

    return (
        <section className="view-panel">
            <h2>Predicción normal</h2>
            <p>
                Analiza la predicción del modelo activo sobre la imagen seleccionada.
            </p>

            <PredictionPanel
                selectedDataset={selectedDataset}
                selectedImage={selectedImage}
            />
        </section>
    )
}

export default PredictionView