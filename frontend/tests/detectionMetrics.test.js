import assert from 'node:assert/strict'
import test from 'node:test'

import {
    findClassMetrics,
    formatBoundingBox,
    formatSignedPercentagePoints,
    formatTidePoints,
    getDetectionUnavailableMessage,
    getObjectOutcomeItems,
    getTideErrorRows,
    mergePrecisionRecallCurves,
} from '../src/utils/detectionMetrics.js'

test('traduce los motivos tipados de indisponibilidad supervisada', () => {
    assert.match(
        getDetectionUnavailableMessage('no_annotation_snapshot'),
        /anotaciones de detección/,
    )
    assert.match(getDetectionUnavailableMessage('unknown'), /no están disponibles/)
})

test('formatea valores TIDE en puntos de AP sin reescalarlos', () => {
    assert.equal(formatTidePoints(3.456), '3.5 pp AP')
    assert.equal(formatTidePoints(2.5, 1, true), '+2.5 pp AP')
    assert.equal(formatTidePoints(-1.25, 1, true), '-1.3 pp AP')
    assert.equal(formatTidePoints(null), '—')
})

test('combina TIDE original, adversarial y cambios por tipo tipado', () => {
    const analysis = {
        original: {
            main_errors: [
                { error_type: 'classification', delta_ap: 1, count: 2 },
                { error_type: 'missed', delta_ap: 3, count: 4 },
            ],
        },
        adversarial: {
            main_errors: [
                { error_type: 'classification', delta_ap: 2.5, count: 3 },
                { error_type: 'missed', delta_ap: 5, count: 7 },
            ],
        },
        changes: [
            { error_type: 'classification', impact_change: 1.5, count_change: 1 },
            { error_type: 'missed', impact_change: 2, count_change: 3 },
        ],
    }

    const rows = getTideErrorRows(analysis)

    assert.deepEqual(rows.map((row) => row.errorType), [
        'classification',
        'missed',
    ])
    assert.equal(rows[0].adversarial.delta_ap, 2.5)
    assert.equal(rows[1].change.count_change, 3)
    assert.match(rows[0].description, /clase incorrecta/)
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
