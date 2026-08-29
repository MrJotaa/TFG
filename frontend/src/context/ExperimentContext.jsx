import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
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
    const [selectedDataset, setSelectedDataset] = useState(null)
    const [selectedImage, setSelectedImage] = useState(null)
    const [activeModel, setActiveModel] = useState(null)
    const [attackCatalog, setAttackCatalog] = useState([])
    const [attackCatalogError, setAttackCatalogError] = useState('')
    const [selectedAttack, setSelectedAttack] = useState('fgsm')
    const selectedAttackRef = useRef(selectedAttack)
    const [attackParameters, setAttackParameters] = useState({})
    const [recoveredEvaluation, setRecoveredEvaluation] = useState(null)
    const [recoveredIndividualAnalysis, setRecoveredIndividualAnalysis] = useState(null)
    const [pendingIndividualComparison, setPendingIndividualComparison] = useState(null)

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
                setAttackParameters((current) =>
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
        setSelectedDataset(dataset)
    }, [clearRecoveredResults])

    const selectImage = useCallback((image) => {
        clearRecoveredResults()
        setSelectedImage(image)
    }, [clearRecoveredResults])

    const selectModel = useCallback((model) => {
        clearRecoveredResults()
        setActiveModel(model)
    }, [clearRecoveredResults])

    const selectAttack = useCallback((attackId) => {
        selectedAttackRef.current = attackId
        clearRecoveredResults()
        setSelectedAttack(attackId)
        setAttackParameters((current) =>
            buildAttackParameters(attackCatalog, attackId, current),
        )
    }, [attackCatalog, clearRecoveredResults])

    const updateAttackParameters = useCallback((parameters) => {
        clearRecoveredResults()
        setAttackParameters(parameters)
    }, [clearRecoveredResults])

    const openRecoveredEvaluation = useCallback(({ historyItem, stored, dataset }) => {
        selectedAttackRef.current = historyItem.attack_type
        setSelectedDataset(dataset)
        setSelectedAttack(historyItem.attack_type)
        setAttackParameters(
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
    }, [attackCatalog])

    const openRecoveredIndividualAnalysis = useCallback(({ historyItem, stored, dataset, image }) => {
        selectedAttackRef.current = historyItem.attack_type
        setSelectedDataset(dataset)
        setSelectedImage(image)
        setSelectedAttack(historyItem.attack_type)
        setAttackParameters(
            buildAttackParameters(
                attackCatalog,
                historyItem.attack_type,
                historyItem.parameters,
            ),
        )
        setPendingIndividualComparison(null)
        setRecoveredEvaluation(null)
        setRecoveredIndividualAnalysis({
            historyItem,
            taskType: historyItem.task_type,
            data: stored.analysis,
        })
    }, [attackCatalog])

    const beginIndividualComparison = useCallback(({ base, kind }) => {
        selectedAttackRef.current = base.item.attack_type
        setSelectedDataset(base.dataset)
        setSelectedImage(base.image)
        setSelectedAttack(base.item.attack_type)
        setAttackParameters(
            buildAttackParameters(
                attackCatalog,
                base.item.attack_type,
                base.item.parameters,
            ),
        )
        setRecoveredEvaluation(null)
        setRecoveredIndividualAnalysis(null)
        setPendingIndividualComparison({ base, kind })
    }, [attackCatalog])

    const clearPendingIndividualComparison = useCallback(() => {
        setPendingIndividualComparison(null)
    }, [])

    const resetExperiment = useCallback(() => {
        selectedAttackRef.current = 'fgsm'
        setDatasets([])
        setSelectedDataset(null)
        setSelectedImage(null)
        setActiveModel(null)
        setSelectedAttack('fgsm')
        setAttackParameters(buildAttackParameters(attackCatalog, 'fgsm'))
        setRecoveredEvaluation(null)
        setRecoveredIndividualAnalysis(null)
        setPendingIndividualComparison(null)
    }, [attackCatalog])

    const value = useMemo(() => ({
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
        pendingIndividualComparison,
        beginIndividualComparison,
        clearPendingIndividualComparison,
        openRecoveredEvaluation,
        openRecoveredIndividualAnalysis,
        resetExperiment,
    }), [
        activeModel, attackCatalog, attackCatalogError, attackCatalogLoading, attackParameters,
        beginIndividualComparison, clearPendingIndividualComparison, datasets,
        openRecoveredEvaluation, openRecoveredIndividualAnalysis, pendingIndividualComparison,
        recoveredEvaluation, recoveredIndividualAnalysis, resetExperiment, selectAttack,
        selectDataset, selectImage, selectModel, selectedAttack, selectedDataset, selectedImage,
        updateAttackParameters,
    ])

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
