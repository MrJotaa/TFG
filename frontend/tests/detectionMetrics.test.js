import assert from 'node:assert/strict'
import test from 'node:test'

import {
    findClassMetrics,
    formatBoundingBox,
    formatSignedPercentagePoints,
    getDetectionUnavailableMessage,
    getObjectOutcomeItems,
    mergePrecisionRecallCurves,
} from '../src/utils/detectionMetrics.js'

test('traduce los motivos tipados de indisponibilidad supervisada', () => {
    assert.match(
        getDetectionUnavailableMessage('no_annotation_snapshot'),
        /snapshot de anotaciones/,
    )
    assert.match(getDetectionUnavailableMessage('unknown'), /no están disponibles/)
})

test('formatea deltas conservando signo y nulos', () => {
    assert.equal(formatSignedPercentagePoints(0.125), '+12.5 pp')
    assert.equal(formatSignedPercentagePoints(-0.04), '-4.0 pp')
    assert.equal(formatSignedPercentagePoints(null), '—')
})

test('construye el desglose de outcomes incluso con campos ausentes', () => {
    assert.deepEqual(
        getObjectOutcomeItems({ preserved_count: 3, suppressed_count: 2 }).map(
            ({ count }) => count,
        ),
        [3, 2, 0, 0],
    )
})

test('fusiona curvas PR por recall sin depender del orden de entrada', () => {
    const merged = mergePrecisionRecallCurves(
        { points: [{ recall: 1, precision: 0.4 }, { recall: 0, precision: 1 }] },
        { points: [{ recall: 0, precision: 0.8 }, { recall: 1, precision: 0.2 }] },
    )

    assert.deepEqual(merged, [
        { recall: 0, original: 1, adversarial: 0.8 },
        { recall: 1, original: 0.4, adversarial: 0.2 },
    ])
})

test('resuelve clases por el ID canónico del modelo', () => {
    const analysis = {
        per_class: [
            { reference: { model_class_id: 4 }, support: 2 },
            { reference: { model_class_id: 9 }, support: 1 },
        ],
    }

    assert.equal(findClassMetrics(analysis, '9').support, 1)
    assert.equal(findClassMetrics(analysis, 3), undefined)
})

test('presenta bounding boxes en el formato público xyxy', () => {
    assert.equal(
        formatBoundingBox({ x1: 1, y1: 2.25, x2: 30, y2: 40.125 }),
        '1.0, 2.3, 30.0, 40.1',
    )
    assert.equal(formatBoundingBox(null), '—')
})
