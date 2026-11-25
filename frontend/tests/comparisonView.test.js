import assert from 'node:assert/strict'
import test from 'node:test'
import { buildComparisonViewState, getIndividualAnalysisTabs } from '../src/utils/comparisonView.js'

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

test('individual analysis blocks model-restricted attacks without hiding historical results', () => {
    const restricted = {
        id: 'detection_localization_pgd',
        supported_task_types: ['detection'],
        supported_model_ids: ['rf_detr_nano'],
    }
    const current = buildComparisonViewState({
        ...base,
        activeModel: { id: 'yolo26n', name: 'YOLO', task_type: 'detection' },
        attackCatalog: [restricted],
        selectedAttack: restricted.id,
    })
    assert.equal(current.historyMode, false)
    assert.equal(current.isAttackSupported, false)

    const recovered = buildComparisonViewState({
        ...base,
        activeModel: { id: 'yolo26n', name: 'YOLO', task_type: 'detection' },
        attackCatalog: [restricted],
        selectedAttack: 'random_noise',
        recoveredIndividualAnalysis: {
            historyItem: {
                analysis_id: 'historical-pgd',
                model_id: 'rf_detr_nano',
                model_name: 'RF-DETR',
                task_type: 'detection',
                attack_type: restricted.id,
                parameters: { epsilon: 0.03 },
            },
            data: {
                result_id: 'stored-pgd',
                original_detections: [],
                adversarial_detections: [],
            },
        },
    })
    assert.equal(recovered.historyMode, true)
    assert.equal(recovered.viewModel.id, 'rf_detr_nano')
    assert.equal(recovered.viewAttack, restricted.id)
    assert.equal(recovered.isAttackSupported, true)
    assert.equal(recovered.attackResult.result_id, 'stored-pgd')
})


test('individual workflows expose exactly two tabs', () => {
    assert.deepEqual(getIndividualAnalysisTabs('prediction'), [
        ['clean', 'Predicción'], ['explanation', 'Explicabilidad'],
    ])
    assert.deepEqual(getIndividualAnalysisTabs('attack'), [
        ['attack', 'Comparación'], ['explanation', 'Explicabilidad'],
    ])
})

for (const task_type of ['classification', 'detection']) {
    test(`${task_type}: individual mode selects only its own panels`, () => {
        const context = { ...base, activeModel: { ...base.activeModel, task_type } }
        const imageContextKey = 'dataset-1:image-1:model-1'
        for (const [mode, initial, excluded] of [['prediction', 'clean', 'attack'], ['attack', 'attack', 'clean']]) {
            const state = buildComparisonViewState({ ...context, mode })
            assert.equal(state.analysisMode, mode)
            assert.equal(state.activePanel, initial)
            assert.equal(buildComparisonViewState({
                ...context, mode, activePanelState: { contextKey: imageContextKey, panel: excluded },
            }).activePanel, initial)
            assert.equal(buildComparisonViewState({
                ...context, mode, activePanelState: { contextKey: imageContextKey, panel: 'explanation' },
            }).activePanel, 'explanation')
            assert.equal(buildComparisonViewState({
                ...context, mode, activePanelState: { contextKey: 'other-image', panel: 'explanation' },
            }).activePanel, initial)
        }
    })
}

test('prediction works without a compatible attack and ignores adversarial results', () => {
    const context = { ...base, attackCatalog: [], selectedAttack: null, attackParameters: {} }
    const attackContextKey = buildComparisonViewState(context).attackContextKey
    const state = buildComparisonViewState({
        ...context, mode: 'prediction',
        attackResultState: { contextKey: attackContextKey, data: { result_id: 'stale', original_prediction: { prediction: 'wrong' } } },
        predictionResultState: { contextKey: 'dataset-1:image-1:model-1', data: { prediction: 'tench' } },
    })
    assert.equal(state.activePanel, 'clean')
    assert.equal(state.isAttackSupported, false)
    assert.equal(state.attackResult, null)
    assert.equal(state.currentReference, null)
    assert.equal(state.cleanPrediction.prediction, 'tench')
    assert.equal(state.hasTemporaryWork, false)
})

test('history opens in comparison and does not leak into the prediction route', () => {
    const recoveredIndividualAnalysis = {
        historyItem: { analysis_id: 'saved', model_id: 'detector', model_name: 'Detector', task_type: 'detection', attack_type: 'square_detection', parameters: {} },
        data: { result_id: 'saved-result', original_detections: [], adversarial_detections: [] },
    }
    const attack = buildComparisonViewState({ ...base, recoveredIndividualAnalysis })
    assert.equal(attack.activePanel, 'attack')
    assert.equal(attack.historyMode, true)
    const prediction = buildComparisonViewState({ ...base, mode: 'prediction', recoveredIndividualAnalysis })
    assert.equal(prediction.activePanel, 'clean')
    assert.equal(prediction.historyMode, false)
    assert.equal(prediction.viewModel.id, base.activeModel.id)
    assert.equal(prediction.attackResult, null)
})

test('preparing B preserves the adversarial mode and temporary base protection', () => {
    const state = buildComparisonViewState({
        ...base, mode: 'prediction', pendingIndividualComparison: { base: { temporary: true } },
    })
    assert.equal(state.analysisMode, 'attack')
    assert.equal(state.activePanel, 'attack')
    assert.equal(state.hasTemporaryWork, true)
})


for (const task_type of ['classification', 'detection']) {
    test(`${task_type}: completed clean prediction and explanation do not require saving`, () => {
        const context = { ...base, mode: 'prediction', activeModel: { ...base.activeModel, task_type } }
        const contextKey = 'dataset-1:image-1:model-1'
        const predictionResultState = { contextKey, data: { prediction: 'tench', detections: [] } }
        const explainabilityResultState = { contextKey, data: { image_url: '/explanation.png' } }
        for (const results of [
            { predictionResultState },
            { explainabilityResultState },
            { predictionResultState, explainabilityResultState },
        ]) {
            const state = buildComparisonViewState({ ...context, ...results })
            assert.equal(state.hasTemporaryWork, false)
            assert.equal(state.busy, false)
        }
        for (const operation of ['loadingPrediction', 'loadingExplainability']) {
            const state = buildComparisonViewState({ ...context, [operation]: true })
            assert.equal(state.hasTemporaryWork, false)
            assert.equal(state.busy, true)
        }
    })
}
