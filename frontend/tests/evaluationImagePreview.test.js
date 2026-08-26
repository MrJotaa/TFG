import assert from 'node:assert/strict'
import test from 'node:test'

import {
    clampEvaluationPreviewPan,
    clampEvaluationPreviewZoom,
    getEvaluationPreviewPanBounds,
    stepEvaluationPreviewZoom,
} from '../src/utils/evaluationImagePreview.js'

test('evaluation preview zoom stays inside the supported interval', () => {
    assert.equal(clampEvaluationPreviewZoom(0.5), 1)
    assert.equal(clampEvaluationPreviewZoom(2.5), 2.5)
    assert.equal(clampEvaluationPreviewZoom(8), 4)
    assert.equal(clampEvaluationPreviewZoom('invalid'), 1)
})

test('evaluation preview zoom steps in both directions and respects limits', () => {
    assert.equal(stepEvaluationPreviewZoom(1, 1), 1.25)
    assert.equal(stepEvaluationPreviewZoom(1.25, -1), 1)
    assert.equal(stepEvaluationPreviewZoom(4, 1), 4)
    assert.equal(stepEvaluationPreviewZoom(1, -1), 1)
})

test('evaluation preview pan bounds only expose the overflow created by zoom', () => {
    assert.deepEqual(
        getEvaluationPreviewPanBounds({
            viewportWidth: 800,
            viewportHeight: 600,
            imageWidth: 700,
            imageHeight: 500,
            zoom: 1,
        }),
        { x: 0, y: 0 },
    )

    assert.deepEqual(
        getEvaluationPreviewPanBounds({
            viewportWidth: 800,
            viewportHeight: 600,
            imageWidth: 700,
            imageHeight: 500,
            zoom: 2,
        }),
        { x: 300, y: 200 },
    )
})

test('evaluation preview pan is clamped so the image cannot be dragged away', () => {
    assert.deepEqual(
        clampEvaluationPreviewPan(
            { x: 420, y: -250 },
            { x: 300, y: 200 },
        ),
        { x: 300, y: -200 },
    )

    assert.deepEqual(
        clampEvaluationPreviewPan(
            { x: -40, y: 60 },
            { x: 300, y: 200 },
        ),
        { x: -40, y: 60 },
    )
})
