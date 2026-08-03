import assert from 'node:assert/strict'
import test from 'node:test'

import {
    CLASSIFICATION_METRIC_GUIDES,
    DETECTION_METRIC_GUIDES,
} from '../src/utils/evaluationMetricGuides.js'

test('clasificación dispone de una guía específica para cada pestaña', () => {
    assert.deepEqual(Object.keys(CLASSIFICATION_METRIC_GUIDES), [
        'overview',
        'classes',
        'matrices',
        'curves',
        'images',
    ])
})

test('detección dispone de una guía específica para cada pestaña', () => {
    assert.deepEqual(Object.keys(DETECTION_METRIC_GUIDES), [
        'overview',
        'coco',
        'classes',
        'curves',
        'objects',
        'tide',
        'images',
    ])
})

test('cada guía explica términos y aporta una clave de interpretación', () => {
    const guides = [
        ...Object.values(CLASSIFICATION_METRIC_GUIDES),
        ...Object.values(DETECTION_METRIC_GUIDES),
    ]

    guides.forEach((guide) => {
        assert.ok(guide.title.length > 10)
        assert.ok(guide.summary.length > 20)
        assert.ok(guide.items.length >= 4)
        assert.ok(guide.items.every((item) => item.term && item.definition.length > 30))
        assert.ok(guide.note.length > 20)
    })
})

test('las guías distinguen las unidades y agregaciones más ambiguas', () => {
    assert.match(DETECTION_METRIC_GUIDES.coco.items.map((item) => item.definition).join(' '), /puntos porcentuales/)
    assert.match(DETECTION_METRIC_GUIDES.tide.items.map((item) => item.term).join(' '), /dAP/)
    assert.match(CLASSIFICATION_METRIC_GUIDES.curves.items.map((item) => item.definition).join(' '), /micro/i)
    assert.match(CLASSIFICATION_METRIC_GUIDES.overview.items.map((item) => item.definition).join(' '), /Macro/)
})
