import assert from 'node:assert/strict'
import test from 'node:test'
import {
    buildIndividualAnalysisComparison,
    comparisonExplanationState,
    formatComparisonMagnitude,
    getComparisonExplanationMessage,
    getComparisonKeyboardAction,
    getComparisonLayerCaption,
    loadIndividualAnalysisComparison,
    comparisonLayerOptions,
    individualAnalysisComparisonSummary,
    resolveIndividualAnalysisComparisonSource,
} from '../src/utils/individualAnalysisComparison.js'

function item(overrides = {}) {
    return {
        analysis_id: 'analysis-a',
        task_type: 'classification',
        dataset_id: 'dataset-1',
        dataset_revision: 4,
        image_id: 'image-1',
        model_id: 'resnet18',
        model_revision: 'm1',
        attack_type: 'fgsm',
        attack_revision: 'a1',
        parameters: { epsilon: 0.03 },
        pipeline_version: 'classification-individual-analysis-v1',
        ...overrides,
    }
}

test('individual comparison accepts one controlled variable at a time', () => {
    const base = item()

    assert.equal(
        buildIndividualAnalysisComparison(
            base,
            item({ analysis_id: 'b', attack_type: 'pgd', attack_revision: 'pgd-1' }),
        ).kind,
        'attack',
    )
    assert.equal(
        buildIndividualAnalysisComparison(
            base,
            item({ analysis_id: 'b', model_id: 'swin', model_revision: 's1' }),
        ).kind,
        'model',
    )
    assert.equal(
        buildIndividualAnalysisComparison(
            base,
            item({ analysis_id: 'b', parameters: { epsilon: 0.05 } }),
        ).kind,
        'attack_configuration',
    )
})

test('individual comparison rejects a different image or multiple changed variables', () => {
    const base = item()

    assert.equal(
        buildIndividualAnalysisComparison(
            base,
            item({ analysis_id: 'b', image_id: 'image-2' }),
        ).compatible,
        false,
    )
    assert.equal(
        buildIndividualAnalysisComparison(
            base,
            item({
                analysis_id: 'b',
                model_id: 'swin',
                model_revision: 's1',
                attack_type: 'pgd',
                attack_revision: 'pgd-1',
            }),
        ).compatible,
        false,
    )
})

test('classification comparison resolves image, perturbation and stored Grad-CAM layers', () => {
    const analysis = {
        adversarial_image_url: '/results/a/adversarial.png',
        perturbation_image_url: '/results/a/perturbation.png',
        original_gradcam_image_url: '/results/a/gradcam-original.png',
        adversarial_gradcam_image_url: '/results/a/gradcam-adversarial.png',
    }
    const image = { url: '/datasets/image.png' }

    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'classification', analysis, image, layer: 'image', stage: 'original',
    }), '/datasets/image.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'classification', analysis, image, layer: 'image', stage: 'adversarial',
    }), '/results/a/adversarial.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'classification', analysis, image, layer: 'perturbation', stage: 'adversarial',
    }), '/results/a/perturbation.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'classification', analysis, image, layer: 'explanation', stage: 'original',
    }), '/results/a/gradcam-original.png')
})

test('detection comparison switches annotated and raw sources without recalculation', () => {
    const analysis = {
        original_annotated_image_url: '/results/a/original-boxes.png',
        adversarial_image_url: '/results/a/adversarial.png',
        adversarial_annotated_image_url: '/results/a/adversarial-boxes.png',
        perturbation_image_url: '/results/a/perturbation.png',
        original_explanation_image_url: '/results/a/eigencam-original.png',
        adversarial_explanation_image_url: '/results/a/eigencam-adversarial.png',
    }
    const image = { url: '/datasets/image.png' }

    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'detection', analysis, image, layer: 'image', stage: 'original', showAnnotations: true,
    }), '/results/a/original-boxes.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'detection', analysis, image, layer: 'image', stage: 'original', showAnnotations: false,
    }), '/datasets/image.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'detection', analysis, image, layer: 'image', stage: 'adversarial', showAnnotations: false,
    }), '/results/a/adversarial.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'detection', analysis, image, layer: 'explanation', stage: 'original',
    }), '/results/a/eigencam-original.png')
    assert.equal(resolveIndividualAnalysisComparisonSource({
        taskType: 'detection', analysis, image, layer: 'explanation', stage: 'adversarial',
    }), '/results/a/eigencam-adversarial.png')
})

test('explanation control is exposed only when both saved analyses have comparable artifacts', () => {
    const complete = {
        original_gradcam_image_url: '/a.png',
        adversarial_gradcam_image_url: '/b.png',
    }
    const incomplete = {
        original_gradcam_image_url: '/a.png',
        adversarial_gradcam_image_url: null,
    }

    assert.deepEqual(comparisonExplanationState('classification', complete, complete), {
        left: true,
        right: true,
        comparable: true,
    })
    assert.equal(
        comparisonLayerOptions('classification', complete, incomplete)
            .some((option) => option.id === 'explanation'),
        false,
    )
})

test('individual summaries expose persisted classification and detection impact', () => {
    const classification = individualAnalysisComparisonSummary('classification', {
        original_prediction: {
            prediction: 'tench',
            top_k: [
                { label: 'tench', probability: 0.8 },
                { label: 'salmon', probability: 0.12 },
                { label: 'trout', probability: 0.05 },
            ],
        },
        adversarial_prediction: {
            prediction: 'crab',
            top_k: [
                { label: 'crab', probability: 0.6 },
                { label: 'tench', probability: 0.2 },
                { label: 'lobster', probability: 0.1 },
            ],
        },
        comparison: {
            prediction_changed: true,
            attack_success: true,
            original_confidence: 0.8,
            adversarial_confidence: 0.2,
            confidence_difference: -0.6,
            perturbation_linf: 0.03,
            perturbation_l2: 4.5,
        },
    })
    assert.equal(classification.primaryLabel, 'tench → crab')
    assert.equal(classification.perturbationLinf, 0.03)
    assert.deepEqual(classification.originalTopK.map((entry) => entry.label), [
        'tench',
        'salmon',
        'trout',
    ])
    assert.deepEqual(classification.adversarialTopK.map((entry) => entry.label), [
        'crab',
        'tench',
        'lobster',
    ])

    const detection = individualAnalysisComparisonSummary('detection', {
        original_detections: [{}, {}, {}],
        adversarial_detections: [{}],
        comparison: {
            attack_success: true,
            average_original_confidence: 0.7,
            average_adversarial_confidence: 0.4,
            average_confidence_difference: 0.3,
            perturbation_linf: 0.02,
            perturbation_l2: 3,
        },
    })
    assert.equal(detection.primaryLabel, '3 → 1')
    assert.equal(detection.primaryCaption, 'detecciones')
})


test('comparison magnitude formatting avoids regex backtracking and trims decimals', () => {
    assert.equal(formatComparisonMagnitude('invalid'), '—')
    assert.equal(formatComparisonMagnitude(0), '0')
    assert.equal(formatComparisonMagnitude(0.00012), '1.20e-4')
    assert.equal(formatComparisonMagnitude(1.23), '1.23')
    assert.equal(formatComparisonMagnitude(4), '4')
})


test('comparison explorer helpers describe layers, explanations and keyboard actions', () => {
    assert.equal(getComparisonLayerCaption('perturbation', 'classification', 'original'), 'Mapa amplificado de la perturbación')
    assert.equal(getComparisonLayerCaption('explanation', 'detection', 'original'), 'Eigen-CAM persistido')
    assert.equal(getComparisonLayerCaption('explanation', 'classification', 'original'), 'Grad-CAM persistido')
    assert.equal(getComparisonLayerCaption('image', 'classification', 'original'), 'Resultado original')
    assert.equal(getComparisonLayerCaption('image', 'classification', 'adversarial'), 'Resultado tras el ataque')

    assert.match(getComparisonExplanationMessage({
        taskType: 'classification',
        comparisonKind: 'model',
        explanationState: { left: false, right: false },
    }), /ambos análisis conservaron su Grad-CAM/)
    assert.match(getComparisonExplanationMessage({
        taskType: 'detection',
        comparisonKind: 'attack',
        explanationState: { left: true, right: false },
    }), /solo uno de los análisis conserva Eigen-CAM/)
    assert.match(getComparisonExplanationMessage({
        taskType: 'classification',
        comparisonKind: 'attack',
        explanationState: { left: false, right: false },
    }), /ninguno de los dos análisis conserva Grad-CAM/)

    assert.deepEqual(getComparisonKeyboardAction('ArrowLeft', 2, 1), { type: 'pan', x: 24, y: 0 })
    assert.equal(getComparisonKeyboardAction('ArrowLeft', 1, 1), null)
    assert.deepEqual(getComparisonKeyboardAction('+', 1, 1), { type: 'zoom', direction: 1 })
    assert.deepEqual(getComparisonKeyboardAction('-', 1, 1), { type: 'zoom', direction: -1 })
    assert.deepEqual(getComparisonKeyboardAction('0', 1, 1), { type: 'reset' })
    assert.equal(getComparisonKeyboardAction('Enter', 1, 1), null)
})

test('comparison loading accepts transient pairs and saved pairs', async () => {
    const leftItem = item()
    const rightItem = item({ analysis_id: 'analysis-b', parameters: { epsilon: 0.04 } })
    const left = { item: leftItem, analysis: { result_id: 'result-a' }, dataset: { id: 'dataset-1' }, image: { id: 'image-1' }, temporary: true }
    const right = { item: rightItem, analysis: { result_id: 'result-b' }, dataset: left.dataset, image: left.image, temporary: true }

    const transient = await loadIndividualAnalysisComparison({
        transientPair: { left, right },
        leftId: null,
        rightId: null,
        getStoredIndividualAnalysis: async () => { throw new Error('not used') },
        getDatasetById: async () => { throw new Error('not used') },
    })
    assert.equal(transient.status, 'ready')
    assert.equal(transient.left, left)

    const stored = new Map([
        ['analysis-a', { item: leftItem, analysis: { result_id: 'result-a' } }],
        ['analysis-b', { item: rightItem, analysis: { result_id: 'result-b' } }],
    ])
    const saved = await loadIndividualAnalysisComparison({
        transientPair: null,
        leftId: 'analysis-a',
        rightId: 'analysis-b',
        getStoredIndividualAnalysis: async (id) => stored.get(id),
        getDatasetById: async () => ({ id: 'dataset-1', images: [{ id: 'image-1', filename: 'fish.jpg' }] }),
    })
    assert.equal(saved.status, 'ready')
    assert.equal(saved.left.temporary, false)
    assert.equal(saved.right.resultId, 'result-b')
})

test('comparison loading rejects incomplete transient and stored contexts', async () => {
    await assert.rejects(() => loadIndividualAnalysisComparison({
        transientPair: { left: {}, right: {} },
        getStoredIndividualAnalysis: async () => null,
        getDatasetById: async () => null,
    }), /dos resultados necesarios/)

    await assert.rejects(() => loadIndividualAnalysisComparison({
        transientPair: null,
        leftId: 'same',
        rightId: 'same',
        getStoredIndividualAnalysis: async () => null,
        getDatasetById: async () => null,
    }), /dos análisis individuales diferentes/)

    const leftStored = { item: item(), analysis: {} }
    const rightStored = { item: item({ analysis_id: 'analysis-b', parameters: { epsilon: 0.04 } }), analysis: {} }
    await assert.rejects(() => loadIndividualAnalysisComparison({
        transientPair: null,
        leftId: 'analysis-a',
        rightId: 'analysis-b',
        getStoredIndividualAnalysis: async (id) => id === 'analysis-a' ? leftStored : rightStored,
        getDatasetById: async () => ({ images: [] }),
    }), /imagen original asociada/)
})
