import PrivateImage from './PrivateImage'
import { useState } from 'react'
import { predictImage } from '../api/predict'
import { generateGradCam } from '../api/explainability'
import { resolveApiUrl } from '../api/client'

function PredictionPanelContent({ selectedDataset, selectedImage }) {
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [gradCam, setGradCam] = useState(null)
    const [gradCamLoading, setGradCamLoading] = useState(false)

    async function handlePredict() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un conjunto de datos y una imagen.')
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

    async function handleGenerateGradCam() {
        if (!selectedDataset || !selectedImage) {
            setError('Debes seleccionar un conjunto de datos y una imagen.')
            return
        }

        try {
            setGradCamLoading(true)
            setError('')
            setGradCam(null)

            const result = await generateGradCam({
                datasetId: selectedDataset.id,
                imageId: selectedImage.id,
            })

            setGradCam(result)
        } catch (err) {
            setError(err.message)
        } finally {
            setGradCamLoading(false)
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

            <button
                type="button"
                onClick={handleGenerateGradCam}
                disabled={!selectedDataset || !selectedImage || gradCamLoading}
            >
                {gradCamLoading ? 'Generando Grad-CAM...' : 'Generar Grad-CAM'}
            </button>

            {gradCam && (
                <div>
                    <h3>Grad-CAM</h3>
                    <p>
                        Clase explicada: <strong>{gradCam.prediction}</strong>
                    </p>
                    <p>
                        Confianza: <strong>{gradCam.confidence}</strong>
                    </p>
                    <PrivateImage
                        src={resolveApiUrl(gradCam.gradcam_image_url)}
                        alt="Grad-CAM"
                        width={300}
                    />
                </div>
            )}
        </div>
    )
}

function PredictionPanel({ selectedDataset, selectedImage }) {
    const selectionKey = `${selectedDataset?.id || 'none'}:${selectedImage?.id || 'none'}`

    return (
        <PredictionPanelContent
            key={selectionKey}
            selectedDataset={selectedDataset}
            selectedImage={selectedImage}
        />
    )
}

export default PredictionPanel
