import test from 'node:test'
import assert from 'node:assert/strict'

import {
    clampDashboardSidebarWidth,
    getExperimentReadiness,
    resolveDashboardSidebarWidthKey,
    resolveScrollOverflow,
} from '../src/utils/dashboard.js'

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


test('dashboard sidebar width is clamped to its supported range', () => {
    assert.equal(clampDashboardSidebarWidth(200), 275)
    assert.equal(clampDashboardSidebarWidth(340), 340)
    assert.equal(clampDashboardSidebarWidth(500), 430)
})

test('dashboard sidebar keyboard resizing resolves arrows and boundaries', () => {
    assert.equal(resolveDashboardSidebarWidthKey('ArrowLeft', 300), 290)
    assert.equal(resolveDashboardSidebarWidthKey('ArrowLeft', 275), 275)
    assert.equal(resolveDashboardSidebarWidthKey('ArrowRight', 420), 430)
    assert.equal(resolveDashboardSidebarWidthKey('ArrowRight', 430), 430)
    assert.equal(resolveDashboardSidebarWidthKey('Home', 350), 275)
    assert.equal(resolveDashboardSidebarWidthKey('End', 350), 430)
    assert.equal(resolveDashboardSidebarWidthKey('Enter', 350), null)
})


test('dashboard blocks both execution flows until invalid attack parameters are corrected', () => {
    const context = {
        selectedDataset: { id: 'dataset-1' },
        selectedImage: { id: 'image-1' },
        activeModel: { id: 'resnet18' },
        attackSupported: true,
    }
    const message = 'Las evaluaciones iniciales deben ser menores o iguales que las máximas.'
    for (const selectedImage of [context.selectedImage, null]) {
        const readiness = getExperimentReadiness({
            ...context, selectedImage, attackParameterError: message,
        })
        assert.equal(readiness.canAnalyze, false)
        assert.equal(readiness.canEvaluate, false)
        assert.equal(readiness.message, message)
    }
    const corrected = getExperimentReadiness({ ...context, attackParameterError: '' })
    assert.equal(corrected.canAnalyze, true)
    assert.equal(corrected.canEvaluate, true)
})
