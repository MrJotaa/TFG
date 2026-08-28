import assert from 'node:assert/strict'
import test from 'node:test'
import {
    makeTemporaryAnalysisItem,
    pendingComparisonReadiness,
    resolveSaveState,
    validatePreparedComparison,
} from '../src/utils/individualAnalysisFlow.js'

function identity(overrides = {}) {
    return {
        task_type: 'classification',
        dataset_id: 'dataset-1',
        dataset_name: 'Imagenette',
        dataset_revision: 3,
        image_id: 'image-1',
        image_filename: 'tench.jpg',
        model_id: 'resnet18',
        model_name: 'ResNet-18',
        model_revision: 'resnet18@1',
        attack_type: 'fgsm',
        attack_revision: 'fgsm@1',
        parameters: { epsilon: 0.03 },
        pipeline_version: 'classification-individual-analysis-v1',
        ...overrides,
    }
}

function baseItem(overrides = {}) {
    return {
        ...makeTemporaryAnalysisItem({
            identity: identity(),
            analysis: { result_id: 'attack_1' },
        }),
        ...overrides,
    }
}

test('temporary references preserve the exact experimental identity returned by preflight', () => {
    const item = makeTemporaryAnalysisItem({
        identity: identity(),
        analysis: { result_id: 'attack_123' },
    })
    assert.equal(item.source_result_id, 'attack_123')
    assert.equal(item.dataset_revision, 3)
    assert.equal(item.model_revision, 'resnet18@1')
    assert.equal(item.attack_revision, 'fgsm@1')
})

test('save state distinguishes the current artefact from an equivalent result already stored', () => {
    assert.equal(resolveSaveState('attack_current', { source_result_id: 'attack_current' }), 'saved')
    assert.equal(resolveSaveState('attack_current', { source_result_id: 'attack_old' }), 'equivalent')
})

test('prepared B must change exactly the requested experimental variable', () => {
    const base = baseItem()
    const modelIdentity = identity({ model_id: 'swin', model_name: 'Swin', model_revision: 'swin@1' })
    assert.equal(validatePreparedComparison({ baseItem: base, expectedKind: 'model', identity: modelIdentity }).valid, true)
    assert.equal(validatePreparedComparison({ baseItem: base, expectedKind: 'attack', identity: modelIdentity }).valid, false)

    const attackIdentity = identity({ attack_type: 'pgd', attack_revision: 'pgd@1', parameters: { epsilon: 0.03, steps: 10 } })
    assert.equal(validatePreparedComparison({ baseItem: base, expectedKind: 'attack', identity: attackIdentity }).valid, true)
})

test('dashboard readiness prevents changing multiple variables while preparing B', () => {
    const base = baseItem()
    const common = {
        baseItem: base,
        selectedDataset: { id: 'dataset-1', dataset_revision: 3 },
        selectedImage: { id: 'image-1' },
        activeModel: { id: 'resnet18', task_type: 'classification' },
        selectedAttack: 'pgd',
        attackParameters: { epsilon: 0.03, steps: 10 },
    }
    assert.equal(pendingComparisonReadiness({ ...common, expectedKind: 'attack' }).ready, true)
    assert.equal(pendingComparisonReadiness({
        ...common,
        expectedKind: 'attack',
        activeModel: { id: 'swin', task_type: 'classification' },
    }).ready, false)
})


test('dashboard readiness blocks a dataset revision changed after A', () => {
    const base = baseItem()
    const status = pendingComparisonReadiness({
        baseItem: base,
        expectedKind: 'attack',
        selectedDataset: { id: 'dataset-1', dataset_revision: 4 },
        selectedImage: { id: 'image-1' },
        activeModel: { id: 'resnet18', task_type: 'classification' },
        selectedAttack: 'pgd',
        attackParameters: { epsilon: 0.03, steps: 10 },
    })
    assert.equal(status.ready, false)
    assert.match(status.message, /misma revisión/i)
})
