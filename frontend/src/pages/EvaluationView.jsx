import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import DetectionEvaluationPanel from '../components/DetectionEvaluationPanel'
import EvaluationPanel from '../components/EvaluationPanel'
import { useExperiment } from '../context/ExperimentContext'
import { resolveExperimentTaskType } from '../utils/experimentTask'
import {
    formatAttackParameters,
    getAttackName,
} from '../utils/attackFormatting'

function ContextItem({ label, value }) {
    return (
        <div className="evaluation-context-item">
            <span>{label}</span>
            <strong title={String(value)}>{value}</strong>
        </div>
    )
}

function EvaluationView() {
    const navigate = useNavigate()
    const [isContextOpen, setIsContextOpen] = useState(false)

    const {
        selectedDataset,
        activeModel,
        attackCatalog,
        selectedAttack,
        attackParameters,
        recoveredEvaluation,
        setRecoveredEvaluation,
    } = useExperiment()

    const taskType = resolveExperimentTaskType({
        activeModel,
        recoveredEvaluation,
    })
    const isDetection = taskType === 'detection'
    const attackName = getAttackName(selectedAttack, attackCatalog)
    const contextPanelId = 'evaluation-context-panel'

    return (
        <section className="evaluation-page modern-evaluation-page">
            <header className="evaluation-page-hero compact">
                <div className="evaluation-page-heading">
                    <span className="evaluation-eyebrow">
                        {isDetection ? 'Detección de objetos' : 'Clasificación de imágenes'}
                    </span>
                    <h2>
                        {isDetection
                            ? 'Evaluación de detección'
                            : 'Evaluación de clasificación'}
                    </h2>
                    <p>
                        {isDetection
                            ? 'Compara el comportamiento original y tras el ataque del detector.'
                            : 'Compara el rendimiento original y tras el ataque del modelo.'}
                    </p>
                </div>

                <div className="evaluation-page-actions">
                    <button
                        type="button"
                        className="secondary-action evaluation-context-toggle"
                        aria-expanded={isContextOpen}
                        aria-controls={contextPanelId}
                        onClick={() => setIsContextOpen((current) => !current)}
                    >
                        <span>Contexto de evaluación</span>
                        <span
                            className="evaluation-context-chevron"
                            aria-hidden="true"
                        >
                            {isContextOpen ? '▴' : '▾'}
                        </span>
                    </button>

                    <button
                        type="button"
                        className="secondary-action evaluation-back-button"
                        onClick={() => navigate('/dashboard')}
                    >
                        Volver al panel principal
                    </button>
                </div>
            </header>

            {isContextOpen && (
                <section
                    id={contextPanelId}
                    className="evaluation-context-strip"
                    aria-label="Contexto del experimento"
                >
                    <ContextItem
                        label="Conjunto de datos"
                        value={selectedDataset?.name || 'Sin seleccionar'}
                    />
                    <ContextItem
                        label="Imágenes"
                        value={selectedDataset?.image_count ?? '—'}
                    />
                    <ContextItem
                        label="Modelo"
                        value={
                            recoveredEvaluation?.historyItem?.model_name ||
                            activeModel?.name ||
                            'Sin modelo activo'
                        }
                    />
                    <ContextItem label="Ataque" value={attackName} />
                    <ContextItem
                        label="Parámetros"
                        value={formatAttackParameters(
                            selectedAttack,
                            attackParameters,
                            attackCatalog,
                        )}
                    />
                </section>
            )}

            {isDetection ? (
                <DetectionEvaluationPanel
                    selectedDataset={selectedDataset}
                    selectedAttack={selectedAttack}
                    attackParameters={attackParameters}
                    attackCatalog={attackCatalog}
                    recoveredEvaluation={recoveredEvaluation}
                    onStartNewEvaluation={() => setRecoveredEvaluation(null)}
                />
            ) : (
                <EvaluationPanel
                    selectedDataset={selectedDataset}
                    selectedAttack={selectedAttack}
                    attackParameters={attackParameters}
                    attackCatalog={attackCatalog}
                    recoveredEvaluation={recoveredEvaluation}
                    onStartNewEvaluation={() => setRecoveredEvaluation(null)}
                />
            )}
        </section>
    )
}

export default EvaluationView
