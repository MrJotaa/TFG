import { useNavigate } from 'react-router-dom'
import DetectionEvaluationPanel from '../components/DetectionEvaluationPanel'
import EvaluationPanel from '../components/EvaluationPanel'
import { useExperiment } from '../context/ExperimentContext'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'

function ContextItem({ label, value }) {
    return (
        <div className="evaluation-context-item">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    )
}

function EvaluationView() {
    const navigate = useNavigate()

    const {
        selectedDataset,
        activeModel,
        selectedAttack,
        attackParameters,
    } = useExperiment()

    const isDetection = activeModel?.task_type === 'detection'
    const attackName = getAttackName(selectedAttack)

    return (
        <section className="evaluation-page modern-evaluation-page">
            <header className="evaluation-page-hero">
                <div>
                    <span className="evaluation-eyebrow">
                        {isDetection ? 'Object detection' : 'Image classification'}
                    </span>
                    <h2>
                        {isDetection
                            ? 'Evaluación adversarial de detección'
                            : 'Evaluación adversarial de clasificación'}
                    </h2>
                    <p>
                        {isDetection
                            ? 'Mide cómo cambia el comportamiento del detector ante perturbaciones adversarias.'
                            : 'Compara rendimiento limpio, robustez, métricas multiclase y curvas antes y después del ataque.'}
                    </p>
                </div>

                <button
                    type="button"
                    className="secondary-action evaluation-back-button"
                    onClick={() => navigate('/dashboard')}
                >
                    Volver al dashboard
                </button>
            </header>

            <section className="evaluation-context-strip" aria-label="Contexto del experimento">
                <ContextItem
                    label="Dataset"
                    value={selectedDataset?.name || 'Sin seleccionar'}
                />
                <ContextItem
                    label="Imágenes"
                    value={selectedDataset?.image_count ?? '—'}
                />
                <ContextItem
                    label="Modelo"
                    value={activeModel?.name || 'Sin modelo activo'}
                />
                <ContextItem label="Ataque" value={attackName} />
                <ContextItem
                    label="Parámetros"
                    value={formatAttackParameters(selectedAttack, attackParameters)}
                />
            </section>

            {isDetection ? (
                <DetectionEvaluationPanel
                    selectedDataset={selectedDataset}
                    selectedAttack={selectedAttack}
                    attackParameters={attackParameters}
                />
            ) : (
                <EvaluationPanel
                    selectedDataset={selectedDataset}
                    selectedAttack={selectedAttack}
                    attackParameters={attackParameters}
                />
            )}
        </section>
    )
}

export default EvaluationView
