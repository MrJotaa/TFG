import { useState } from 'react'
import { predictImage } from '../api/predict'

function PredictionPanel({ selectedDataset, selectedImage }) {
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    async function handlePredict() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un dataset y una imagen.')
            return
        }

        try {
            setLoading(true)
            setError('')
            setPrediction(null)

            const result = await predictImage({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
            })

            setPrediction(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <h2>Predicción</h2>

            {!selectedImage ? (
                <p>Selecciona una imagen para realizar la predicción.</p>
            ) : (
                <p>
                    Imagen seleccionada: <strong>{selectedImage.filename}</strong>
                </p>
            )}

            {error && <p className="error">{error}</p>}

            <button
                type="button"
                onClick={handlePredict}
                disabled={!selectedDataset || !selectedImage || loading}
            >
                {loading ? 'Prediciendo...' : 'Realizar predicción'}
            </button>

            {prediction && (
                <div>
                    <h3>Resultado</h3>
                    <p>
                        Clase predicha: <strong>{prediction.prediction}</strong>
                    </p>
                    <p>
                        Confianza: <strong>{prediction.confidence}</strong>
                    </p>

                    <h4>Top-K</h4>
                    <ul>
                        {prediction.top_k.map((item) => (
                            <li key={item.label}>
                                {item.label}: {item.probability}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    )
}

export default PredictionPanel