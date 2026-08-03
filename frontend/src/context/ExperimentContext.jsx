import { createContext, useContext, useState } from 'react'

const ExperimentContext = createContext(null)

export function ExperimentProvider({ children }) {
    const [datasets, setDatasets] = useState([])
    const [selectedDataset, setSelectedDataset] = useState(null)
    const [selectedImage, setSelectedImage] = useState(null)
    const [activeModel, setActiveModel] = useState(null)
    const [selectedAttack, setSelectedAttack] = useState('fgsm')
    const [attackParameters, setAttackParameters] = useState({
        epsilon: 0.03,
        alpha: 0.01,
        steps: 10,
    })
    const [recoveredEvaluation, setRecoveredEvaluation] = useState(null)

    function selectDataset(dataset) {
        setRecoveredEvaluation(null)
        setSelectedDataset(dataset)
    }

    function selectModel(model) {
        setRecoveredEvaluation(null)
        setActiveModel(model)
    }

    function selectAttack(attack) {
        setRecoveredEvaluation(null)
        setSelectedAttack(attack)
    }

    function updateAttackParameters(parameters) {
        setRecoveredEvaluation(null)
        setAttackParameters(parameters)
    }

    function openRecoveredEvaluation({ historyItem, stored, dataset }) {
        setSelectedDataset(dataset)
        setSelectedAttack(historyItem.attack_type)
        setAttackParameters(historyItem.parameters)
        setRecoveredEvaluation({
            historyItem,
            taskType: stored.task_type,
            data: stored.evaluation,
        })
    }

    function resetExperiment() {
        setDatasets([])
        setSelectedDataset(null)
        setSelectedImage(null)
        setActiveModel(null)
        setSelectedAttack('fgsm')
        setAttackParameters({
            epsilon: 0.03,
            alpha: 0.01,
            steps: 10,
        })
        setRecoveredEvaluation(null)
    }

    const value = {
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset: selectDataset,
        selectedImage,
        setSelectedImage,
        activeModel,
        setActiveModel: selectModel,
        selectedAttack,
        setSelectedAttack: selectAttack,
        attackParameters,
        setAttackParameters: updateAttackParameters,
        recoveredEvaluation,
        setRecoveredEvaluation,
        openRecoveredEvaluation,
        resetExperiment,
    }

    return (
        <ExperimentContext.Provider value={value}>
            {children}
        </ExperimentContext.Provider>
    )
}

// Context hooks intentionally live beside their provider.
// eslint-disable-next-line react-refresh/only-export-components
export function useExperiment() {
    const context = useContext(ExperimentContext)

    if (!context) {
        throw new Error('useExperiment must be used inside ExperimentProvider')
    }

    return context
}
