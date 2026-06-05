import { useNavigate } from 'react-router-dom'
import EvaluationPanel from '../components/EvaluationPanel'
import { useExperiment } from '../context/ExperimentContext'
import DetectionEvaluationPanel from '../components/DetectionEvaluationPanel'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'

function EvaluationView() {
    const navigate = useNavigate()

    const {
        selectedDataset,
        activeModel,
        selectedAttack,
        attackParameters,
    } = useExperiment()

    const attackName = getAttackName(selectedAttack)

    return (
        <section className="evaluation-page">
            <div className="comparison-header">
                <div>
                    <h2>
                        {activeModel?.task_type === 'detection'
                            ? 'Evaluación y Análisis de Detección'
                            : 'Evaluación y Análisis de Dataset'}
                    </h2>
                    <p>
                        {activeModel?.task_type === 'detection'
                            ? 'Evalúa el impacto de ataques adversarios sobre las detecciones del modelo.'
                            : 'Evalúa el impacto del ataque adversario sobre todas las imágenes del dataset seleccionado.'}
                    </p>
                </div>

                <button
                    type="button"
                    className="secondary-action"
                    onClick={() => navigate('/dashboard')}
                >
                    Volver al Dashboard
                </button>
            </div>

            <div className="context-summary">
                <h3>Resumen del Contexto</h3>

                <div className="context-grid">
                    <p>
                        <strong>Dataset:</strong>{' '}
                        {selectedDataset?.name || 'Sin dataset seleccionado'}
                    </p>

                    <p>
                        <strong>Imágenes Totales:</strong>{' '}
                        {selectedDataset?.image_count ?? '-'}
                    </p>

                    <p>
                        <strong>Modelo:</strong>{' '}
                        {activeModel?.name || 'Sin modelo activo'}
                    </p>

                    <p>
                        <strong>Ataque:</strong> {attackName}
                    </p>

                    <p>
                        <strong>Parámetros:</strong>{' '}
                        {formatAttackParameters(selectedAttack, attackParameters)}
                    </p>
                </div>
            </div>

            {activeModel?.task_type === 'detection' ? (
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