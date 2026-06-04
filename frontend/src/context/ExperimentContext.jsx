import { createContext, useContext, useState } from 'react'

const ExperimentContext = createContext(null)

export function ExperimentProvider({ children }) {
    const [selectedUser, setSelectedUser] = useState(null)
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

    const value = {
        selectedUser,
        setSelectedUser,
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset,
        selectedImage,
        setSelectedImage,
        activeModel,
        setActiveModel,
        selectedAttack,
        setSelectedAttack,
        attackParameters,
        setAttackParameters,
    }

    return (
        <ExperimentContext.Provider value={value}>
            {children}
        </ExperimentContext.Provider>
    )
}

export function useExperiment() {
    const context = useContext(ExperimentContext)

    if (!context) {
        throw new Error('useExperiment must be used inside ExperimentProvider')
    }

    return context
}