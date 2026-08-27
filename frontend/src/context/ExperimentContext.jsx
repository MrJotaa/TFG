import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useRef,
    useState,
} from 'react'
import { getAttacks } from '../api/attacks'
import { useAuth } from './AuthContext'
import { buildAttackParameters } from '../utils/attackCatalog'

const ExperimentContext = createContext(null)

export function ExperimentProvider({ children }) {
    const { isAuthenticated } = useAuth()
    const [datasets, setDatasets] = useState([])
    const [selectedDataset, setSelectedDatasetState] = useState(null)
    const [selectedImage, setSelectedImageState] = useState(null)
    const [activeModel, setActiveModelState] = useState(null)
    const [attackCatalog, setAttackCatalog] = useState([])
    const [attackCatalogError, setAttackCatalogError] = useState('')
    const [selectedAttack, setSelectedAttackState] = useState('fgsm')
    const selectedAttackRef = useRef(selectedAttack)
    const [attackParameters, setAttackParametersState] = useState({})
    const [recoveredEvaluation, setRecoveredEvaluation] = useState(null)
    const [recoveredIndividualAnalysis, setRecoveredIndividualAnalysis] = useState(null)

    const attackCatalogLoading = Boolean(
        isAuthenticated && attackCatalog.length === 0 && !attackCatalogError,
    )

    useEffect(() => {
        if (!isAuthenticated || attackCatalog.length > 0) return undefined

        let cancelled = false

        Promise.resolve().then(() => {
            if (!cancelled) setAttackCatalogError('')
        })

        getAttacks()
            .then((catalog) => {
                if (cancelled) return

                setAttackCatalog(catalog)
                setAttackParametersState((current) =>
                    buildAttackParameters(
                        catalog,
                        selectedAttackRef.current,
                        current,
                    ),
                )
            })
            .catch((error) => {
                if (!cancelled) {
                    setAttackCatalogError(
                        error?.status
                            ? error.message
                            : 'No se pudo conectar con el servidor.',
                    )
                }
            })

        return () => {
            cancelled = true
        }
    }, [isAuthenticated, attackCatalog.length])

    const clearRecoveredResults = useCallback(() => {
        setRecoveredEvaluation(null)
        setRecoveredIndividualAnalysis(null)
    }, [])

    const selectDataset = useCallback((dataset) => {
        clearRecoveredResults()
        setSelectedDatasetState(dataset)
    }, [clearRecoveredResults])

    const selectImage = useCallback((image) => {
        clearRecoveredResults()
        setSelectedImageState(image)
    }, [clearRecoveredResults])

    const selectModel = useCallback((model) => {
        clearRecoveredResults()
        setActiveModelState(model)
    }, [clearRecoveredResults])

    const selectAttack = useCallback((attackId) => {
        selectedAttackRef.current = attackId
        clearRecoveredResults()
        setSelectedAttackState(attackId)
        setAttackParametersState((current) =>
            buildAttackParameters(attackCatalog, attackId, current),
        )
    }, [attackCatalog, clearRecoveredResults])

    const updateAttackParameters = useCallback((parameters) => {
        clearRecoveredResults()
        setAttackParametersState(parameters)
    }, [clearRecoveredResults])

    function openRecoveredEvaluation({ historyItem, stored, dataset }) {
        selectedAttackRef.current = historyItem.attack_type
        setSelectedDatasetState(dataset)
        setSelectedAttackState(historyItem.attack_type)
        setAttackParametersState(
            buildAttackParameters(
                attackCatalog,
                historyItem.attack_type,
                historyItem.parameters,
            ),
        )
        setRecoveredIndividualAnalysis(null)
        setRecoveredEvaluation({
            historyItem,
            taskType: stored.task_type,
            data: stored.evaluation,
        })
    }

    function openRecoveredIndividualAnalysis({ historyItem, stored, dataset, image }) {
        setSelectedDatasetState(dataset)
        setSelectedImageState(image)
        setRecoveredEvaluation(null)
        setRecoveredIndividualAnalysis({
            historyItem,
            taskType: historyItem.task_type,
            data: stored.analysis,
        })
    }

    function resetExperiment() {
        selectedAttackRef.current = 'fgsm'
        setDatasets([])
        setSelectedDatasetState(null)
        setSelectedImageState(null)
        setActiveModelState(null)
        setSelectedAttackState('fgsm')
        setAttackParametersState(buildAttackParameters(attackCatalog, 'fgsm'))
        setRecoveredEvaluation(null)
        setRecoveredIndividualAnalysis(null)
    }

    const value = {
        datasets,
        setDatasets,
        selectedDataset,
        setSelectedDataset: selectDataset,
        selectedImage,
        setSelectedImage: selectImage,
        activeModel,
        setActiveModel: selectModel,
        attackCatalog,
        attackCatalogLoading,
        attackCatalogError,
        selectedAttack,
        setSelectedAttack: selectAttack,
        attackParameters,
        setAttackParameters: updateAttackParameters,
        recoveredEvaluation,
        setRecoveredEvaluation,
        recoveredIndividualAnalysis,
        setRecoveredIndividualAnalysis,
        openRecoveredEvaluation,
        openRecoveredIndividualAnalysis,
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
