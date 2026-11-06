import { useState } from 'react'
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
    const [isContextOpen, setIsContextOpen] = useState(false)
    const [detectionResultContext, setDetectionResultContext] = useState(null)

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
    const modelName = recoveredEvaluation?.historyItem?.model_name || activeModel?.name || 'Sin modelo activo'
    const datasetName = selectedDataset?.name || 'Sin conjunto de datos'
    const contextPanelId = 'evaluation-context-panel'
    const recoveredDetectionResultContext = isDetection && recoveredEvaluation?.data
        ? {
            resultId: recoveredEvaluation.data.result_id,
            datasetRevision: recoveredEvaluation.data.dataset_revision,
        }
        : null
    const activeDetectionResultContext = (
        detectionResultContext ?? recoveredDetectionResultContext
    )

    return (
        <section
            className={`evaluation-page modern-evaluation-page ${
                isDetection
                    ? 'detection-evaluation-page'
                    : 'classification-evaluation-page'
            }`}
        >
            <header className="evaluation-page-hero compact evaluation-page-hero-flat">
                <div className="evaluation-page-heading">
                    <span className="evaluation-eyebrow">
                        {isDetection ? 'Evaluación · detección' : 'Evaluación · clasificación'}
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

                <div className="evaluation-hero-context">
                    <div className="evaluation-context-summary" aria-label="Resumen de configuración">
                        <strong>{datasetName}</strong>
                        <span aria-hidden="true">/</span>
                        <span title={modelName}>{modelName}</span>
                        <span aria-hidden="true">/</span>
                        <span>{attackName}</span>
                    </div>
                    <button
                        type="button"
                        className="secondary-action evaluation-context-toggle"
                        aria-expanded={isContextOpen}
                        aria-controls={contextPanelId}
                        onClick={() => setIsContextOpen((current) => !current)}
                    >
                        <span>Contexto</span>
                        <span className="evaluation-context-chevron" aria-hidden="true">▾</span>
                    </button>
                </div>
            </header>

            <div
                className={`evaluation-context-collapse${isContextOpen ? ' open' : ''}`}
                aria-hidden={!isContextOpen}
            >
                <div className="evaluation-context-collapse-inner">
                    <section
                        id={contextPanelId}
                        className="evaluation-context-strip"
                        aria-label="Contexto del experimento"
                    >
                        <ContextItem label="Conjunto de datos" value={datasetName} />
                        <ContextItem
                            label="Imágenes"
                            value={selectedDataset?.image_count ?? '—'}
                        />
                        <ContextItem label="Modelo" value={modelName} />
                        <ContextItem label="Ataque" value={attackName} />
                        <ContextItem
                            label="Parámetros"
                            value={formatAttackParameters(
                                selectedAttack,
                                attackParameters,
                                attackCatalog,
                            )}
                        />
                        {isDetection && activeDetectionResultContext?.resultId && (
                            <ContextItem
                                label="Resultado"
                                value={`${activeDetectionResultContext.resultId} · revisión ${activeDetectionResultContext.datasetRevision ?? '—'}`}
                            />
                        )}
                    </section>
                </div>
            </div>

            {isDetection ? (
                <DetectionEvaluationPanel
                    selectedDataset={selectedDataset}
                    selectedAttack={selectedAttack}
                    modelId={activeModel?.id}
                    attackParameters={attackParameters}
                    attackCatalog={attackCatalog}
                    recoveredEvaluation={recoveredEvaluation}
                    onStartNewEvaluation={() => setRecoveredEvaluation(null)}
                    onResultContextChange={setDetectionResultContext}
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
