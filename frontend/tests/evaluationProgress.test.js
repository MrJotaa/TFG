import assert from 'node:assert/strict'
import test from 'node:test'

import {
    evaluationProgressMessage,
    evaluationProgressPercent,
} from '../src/utils/evaluationProgress.js'

test('evaluation progress reflects processed images', () => {
    const progress = { processed: 25, total: 100, phase: 'images' }

    assert.equal(evaluationProgressPercent(progress), 25)
    assert.equal(
        evaluationProgressMessage(progress),
        '25 de 100 imágenes procesadas.',
    )
})

test('evaluation progress identifies the final metrics stage', () => {
    const progress = { processed: 100, total: 100, phase: 'metrics' }

    assert.equal(evaluationProgressPercent(progress), 100)
    assert.equal(
        evaluationProgressMessage(progress),
        '100 de 100 imágenes procesadas. Calculando métricas finales.',
    )
})

test('evaluation progress clamps invalid processed counts', () => {
    assert.equal(
        evaluationProgressPercent({ processed: 150, total: 100, phase: 'images' }),
        100,
    )
    assert.equal(evaluationProgressPercent(null), 0)
})
