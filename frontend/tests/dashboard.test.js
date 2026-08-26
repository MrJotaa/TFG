import test from 'node:test'
import assert from 'node:assert/strict'

import { getExperimentReadiness, resolveScrollOverflow } from '../src/utils/dashboard.js'

test('dashboard requires an active model before enabling experiment actions', () => {
    const readiness = getExperimentReadiness({
        selectedDataset: { id: 'dataset-1' },
        selectedImage: { id: 'image-1' },
        activeModel: null,
    })

    assert.equal(readiness.canEvaluate, false)
    assert.equal(readiness.canAnalyze, false)
    assert.equal(readiness.tone, 'idle')
})

test('dashboard blocks actions when the selected attack is incompatible', () => {
    const readiness = getExperimentReadiness({
        selectedDataset: { id: 'dataset-1' },
        selectedImage: { id: 'image-1' },
        activeModel: { id: 'rf-detr', task_type: 'detection' },
        attackSupported: false,
    })

    assert.equal(readiness.canEvaluate, false)
    assert.equal(readiness.canAnalyze, false)
    assert.match(readiness.message, /ataque compatible/i)
})

test('dashboard can evaluate a dataset without selecting an individual image', () => {
    const readiness = getExperimentReadiness({
        selectedDataset: { id: 'dataset-1' },
        selectedImage: null,
        activeModel: { id: 'resnet18' },
        attackSupported: true,
    })

    assert.equal(readiness.canEvaluate, true)
    assert.equal(readiness.canAnalyze, false)
    assert.equal(readiness.tone, 'partial')
})

test('dashboard enables both actions when model, attack, dataset and image are ready', () => {
    const readiness = getExperimentReadiness({
        selectedDataset: { id: 'dataset-1' },
        selectedImage: { id: 'image-1' },
        activeModel: { id: 'resnet18' },
        attackSupported: true,
    })

    assert.equal(readiness.canEvaluate, true)
    assert.equal(readiness.canAnalyze, true)
    assert.equal(readiness.tone, 'ready')
})


test('scroll overflow exposes only the directions that still contain attacks', () => {
    assert.deepEqual(
        resolveScrollOverflow({ scrollTop: 0, scrollHeight: 500, clientHeight: 200 }),
        { canScrollUp: false, canScrollDown: true },
    )
    assert.deepEqual(
        resolveScrollOverflow({ scrollTop: 150, scrollHeight: 500, clientHeight: 200 }),
        { canScrollUp: true, canScrollDown: true },
    )
    assert.deepEqual(
        resolveScrollOverflow({ scrollTop: 300, scrollHeight: 500, clientHeight: 200 }),
        { canScrollUp: true, canScrollDown: false },
    )
})
