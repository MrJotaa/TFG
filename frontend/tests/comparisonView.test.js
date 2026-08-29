import assert from 'node:assert/strict'
import test from 'node:test'
import { buildComparisonViewState } from '../src/utils/comparisonView.js'

const base = {
    selectedDataset: { id: 'dataset-1' },
    selectedImage: { id: 'image-1' },
    activeModel: { id: 'model-1', name: 'ResNet', task_type: 'classification' },
    attackCatalog: [{ id: 'fgsm', supported_task_types: ['classification'] }],
    selectedAttack: 'fgsm',
    attackParameters: { epsilon: 0.03 },
    recoveredIndividualAnalysis: null,
    pendingIndividualComparison: null,
    predictionResultState: null,
    attackResultState: null,
    explainabilityResultState: null,
    activePanelState: null,
    savedAnalysisState: null,
    errorState: null,
    loading: false,
    loadingPrediction: false,
    loadingExplainability: false,
    checkingHistory: false,
    savingAnalysis: false,
}

test('comparison view keeps current classification results scoped to its context', () => {
    const contextKey = 'dataset-1:image-1:model-1'
    const state = buildComparisonViewState({
        ...base,
        predictionResultState: {
            contextKey,
            data: { prediction: 'tench', confidence: 0.9 },
        },
        activePanelState: { contextKey, panel: 'explanation' },
    })

    assert.equal(state.historyMode, false)
    assert.equal(state.isDetection, false)
    assert.equal(state.cleanPrediction.prediction, 'tench')
    assert.equal(state.activePanel, 'explanation')
    assert.equal(state.isAttackSupported, true)
    assert.equal(state.hasTemporaryWork, true)
})

test('comparison view ignores results from another image context', () => {
    const state = buildComparisonViewState({
        ...base,
        predictionResultState: {
            contextKey: 'dataset-1:other:model-1',
            data: { prediction: 'wrong' },
        },
        errorState: { contextKey: 'other', message: 'stale' },
    })

    assert.equal(state.predictionResult, null)
    assert.equal(state.cleanPrediction, null)
    assert.equal(state.error, '')
    assert.equal(state.hasTemporaryWork, false)
})

test('saved history restores the persisted model, attack and reference', () => {
    const state = buildComparisonViewState({
        ...base,
        recoveredIndividualAnalysis: {
            historyItem: {
                analysis_id: 'analysis-1',
                model_id: 'detector-1',
                model_name: 'Detector',
                task_type: 'detection',
                attack_type: 'pgd',
                parameters: { epsilon: 0.1 },
            },
            data: {
                result_id: 'result-1',
                original_detections: [],
                adversarial_detections: [],
            },
        },
    })

    assert.equal(state.historyMode, true)
    assert.equal(state.isDetection, true)
    assert.equal(state.viewModel.id, 'detector-1')
    assert.equal(state.viewAttack, 'pgd')
    assert.equal(state.attackResult.result_id, 'result-1')
    assert.equal(state.saveState, 'saved')
    assert.equal(state.currentReference.temporary, false)
    assert.equal(state.hasTemporaryWork, false)
})

test('temporary attack results build a temporary comparison reference', () => {
    const first = buildComparisonViewState(base)
    const state = buildComparisonViewState({
        ...base,
        attackResultState: {
            contextKey: first.attackContextKey,
            identity: { result_id: 'identity-1' },
            data: {
                result_id: 'result-1',
                original_prediction: { prediction: 'tench' },
            },
        },
    })

    assert.equal(state.attackResult.result_id, 'result-1')
    assert.equal(state.currentReference.temporary, true)
    assert.equal(state.saveState, 'temporary')
    assert.equal(state.hasTemporaryWork, true)
})

test('busy state includes every asynchronous operation', () => {
    const state = buildComparisonViewState({ ...base, savingAnalysis: true })
    assert.equal(state.busy, true)
})
