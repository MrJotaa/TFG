import { resolveCleanDetectionResult } from './adversarialExplorer.js'
import {
    attackParametersKey,
    isAttackSupportedForTask,
} from './attackCatalog.js'
import {
    normalizeClassificationPanel,
    normalizeDetectionPanel,
    resolveClassificationExplanation,
    resolveDetectionExplanation,
} from './individualAnalysis.js'
import {
    makeAnalysisReference,
    makeTemporaryAnalysisItem,
} from './individualAnalysisFlow.js'

function resolveViewContext({
    recoveredIndividualAnalysis,
    activeModel,
    selectedAttack,
    attackParameters,
}) {
    const recoveredHistoryItem = recoveredIndividualAnalysis?.historyItem ?? null
    const recoveredAnalysis = recoveredIndividualAnalysis?.data ?? null
    const historyMode = Boolean(recoveredHistoryItem && recoveredAnalysis)

    if (!historyMode) {
        return {
            recoveredHistoryItem,
            recoveredAnalysis,
            historyMode,
            viewModel: activeModel,
            viewAttack: selectedAttack,
            viewAttackParameters: attackParameters,
        }
    }

    return {
        recoveredHistoryItem,
        recoveredAnalysis,
        historyMode,
        viewModel: {
            id: recoveredHistoryItem.model_id,
            name: recoveredHistoryItem.model_name,
            task_type: recoveredHistoryItem.task_type,
        },
        viewAttack: recoveredHistoryItem.attack_type,
        viewAttackParameters: recoveredHistoryItem.parameters,
    }
}

function resolveScopedData(state, contextKey) {
    if (state?.contextKey !== contextKey) {
        return null
    }
    return state.data
}

function resolveAttackState({
    historyMode,
    recoveredAnalysis,
    attackResultState,
    attackContextKey,
}) {
    if (historyMode) {
        return { attackResult: recoveredAnalysis, attackIdentity: null }
    }
    if (attackResultState?.contextKey !== attackContextKey) {
        return { attackResult: null, attackIdentity: null }
    }
    return {
        attackResult: attackResultState.data,
        attackIdentity: attackResultState.identity ?? null,
    }
}

function resolvePanel({ activePanelState, imageContextKey, historyMode, isDetection }) {
    let requestedPanel = historyMode ? 'attack' : 'clean'
    if (activePanelState?.contextKey === imageContextKey) {
        requestedPanel = activePanelState.panel
    }
    return isDetection
        ? normalizeDetectionPanel(requestedPanel)
        : normalizeClassificationPanel(requestedPanel)
}

function resolveOutputs({
    isDetection,
    predictionResult,
    attackResult,
    explainabilityResult,
}) {
    if (isDetection) {
        return {
            cleanPrediction: null,
            cleanDetection: resolveCleanDetectionResult({
                detectionResult: predictionResult,
                attackResult,
            }),
            classificationExplanation: null,
            detectionExplanation: resolveDetectionExplanation({
                attackResult,
                explainabilityResult,
            }),
        }
    }

    return {
        cleanPrediction: attackResult?.original_prediction ?? predictionResult,
        cleanDetection: null,
        classificationExplanation: resolveClassificationExplanation({
            attackResult,
            explainabilityResult,
        }),
        detectionExplanation: null,
    }
}

function resolveSavedState({ historyMode, savedAnalysisState, attackResult }) {
    if (historyMode) {
        return { currentSavedAnalysis: null, saveState: 'saved' }
    }
    const currentSavedAnalysis = savedAnalysisState?.resultId === attackResult?.result_id
        ? savedAnalysisState
        : null
    return {
        currentSavedAnalysis,
        saveState: currentSavedAnalysis?.status ?? 'temporary',
    }
}

function resolveCurrentReference({
    historyMode,
    recoveredHistoryItem,
    recoveredAnalysis,
    attackResult,
    attackIdentity,
    currentSavedAnalysis,
    saveState,
    selectedDataset,
    selectedImage,
}) {
    if (historyMode) {
        return makeAnalysisReference({
            item: recoveredHistoryItem,
            analysis: recoveredAnalysis,
            dataset: selectedDataset,
            image: selectedImage,
            temporary: false,
        })
    }
    if (!attackResult || !attackIdentity) {
        return null
    }

    const temporaryItem = makeTemporaryAnalysisItem({
        identity: attackIdentity,
        analysis: attackResult,
    })
    const item = saveState === 'saved' && currentSavedAnalysis?.item
        ? currentSavedAnalysis.item
        : temporaryItem

    return makeAnalysisReference({
        item,
        analysis: attackResult,
        dataset: selectedDataset,
        image: selectedImage,
        temporary: saveState !== 'saved',
    })
}

function resolveTemporaryWork({
    pendingIndividualComparison,
    historyMode,
    attackResult,
    saveState,
    predictionResult,
    explainabilityResult,
}) {
    if (pendingIndividualComparison?.base?.temporary) {
        return true
    }
    if (historyMode) {
        return false
    }
    if (attackResult) {
        return saveState !== 'saved'
    }
    return Boolean(predictionResult || explainabilityResult)
}

export function buildComparisonViewState({
    selectedDataset,
    selectedImage,
    activeModel,
    attackCatalog,
    selectedAttack,
    attackParameters,
    recoveredIndividualAnalysis,
    pendingIndividualComparison,
    predictionResultState,
    attackResultState,
    explainabilityResultState,
    activePanelState,
    savedAnalysisState,
    errorState,
    loading,
    loadingPrediction,
    loadingExplainability,
    checkingHistory,
    savingAnalysis,
}) {
    const view = resolveViewContext({
        recoveredIndividualAnalysis,
        activeModel,
        selectedAttack,
        attackParameters,
    })
    const taskType = view.viewModel?.task_type ?? 'classification'
    const isDetection = taskType === 'detection'
    const imageContextKey = [
        selectedDataset?.id,
        selectedImage?.id,
        view.viewModel?.id,
    ].join(':')
    const attackContextKey = [
        imageContextKey,
        view.viewAttack,
        attackParametersKey(view.viewAttackParameters),
    ].join(':')
    const predictionResult = resolveScopedData(predictionResultState, imageContextKey)
    const explainabilityResult = resolveScopedData(explainabilityResultState, imageContextKey)
    const { attackResult, attackIdentity } = resolveAttackState({
        historyMode: view.historyMode,
        recoveredAnalysis: view.recoveredAnalysis,
        attackResultState,
        attackContextKey,
    })
    const activePanel = resolvePanel({
        activePanelState,
        imageContextKey,
        historyMode: view.historyMode,
        isDetection,
    })
    const outputs = resolveOutputs({
        isDetection,
        predictionResult,
        attackResult,
        explainabilityResult,
    })
    const saved = resolveSavedState({
        historyMode: view.historyMode,
        savedAnalysisState,
        attackResult,
    })
    const currentReference = resolveCurrentReference({
        historyMode: view.historyMode,
        recoveredHistoryItem: view.recoveredHistoryItem,
        recoveredAnalysis: view.recoveredAnalysis,
        attackResult,
        attackIdentity,
        currentSavedAnalysis: saved.currentSavedAnalysis,
        saveState: saved.saveState,
        selectedDataset,
        selectedImage,
    })
    const error = errorState?.contextKey === imageContextKey ? errorState.message : ''
    const busy = Boolean(
        loading || loadingPrediction || loadingExplainability || checkingHistory || savingAnalysis,
    )
    const hasTemporaryWork = resolveTemporaryWork({
        pendingIndividualComparison,
        historyMode: view.historyMode,
        attackResult,
        saveState: saved.saveState,
        predictionResult,
        explainabilityResult,
    })

    return {
        ...view,
        taskType,
        isDetection,
        isAttackSupported: isAttackSupportedForTask(
            attackCatalog,
            view.viewAttack,
            taskType,
        ),
        imageContextKey,
        attackContextKey,
        predictionResult,
        explainabilityResult,
        attackResult,
        attackIdentity,
        activePanel,
        ...outputs,
        ...saved,
        currentReference,
        error,
        busy,
        hasTemporaryWork,
    }
}
