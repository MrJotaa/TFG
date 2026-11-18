import assert from 'node:assert/strict'
import test from 'node:test'

import {
    CLASSIFICATION_METRIC_GUIDES,
    CLASSIFICATION_METRIC_HELP,
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

test('cada guía explica cómo interpretar su pestaña y aporta una clave final', () => {
    const guides = [
        ...Object.values(CLASSIFICATION_METRIC_GUIDES),
        ...Object.values(DETECTION_METRIC_GUIDES),
    ]

    guides.forEach((guide) => {
        assert.ok(guide.title.length > 10)
        assert.ok(guide.summary.length > 20)
        assert.ok(guide.items.length >= 3)
        assert.ok(guide.items.every((item) => item.term && item.definition.length > 30))
        assert.ok(guide.note.length > 20)
    })
})

test('la guía de clasificación prioriza interpretación antes que fórmulas aisladas', () => {
    assert.match(CLASSIFICATION_METRIC_GUIDES.overview.items[0].definition, /referencia|resistencia/i)
    assert.match(CLASSIFICATION_METRIC_GUIDES.overview.items[1].definition, /ASR condicional/i)
    assert.match(CLASSIFICATION_METRIC_GUIDES.images.items[2].definition, /causalidad/i)
})

test('las ayudas contextuales explican agregaciones y métricas ambiguas', () => {
    assert.match(CLASSIFICATION_METRIC_HELP.micro, /todas las clases/i)
    assert.match(CLASSIFICATION_METRIC_HELP.macro, /mismo peso|peso/i)
    assert.match(CLASSIFICATION_METRIC_HELP.conditional_asr, /correctamente antes del ataque/i)
    assert.match(CLASSIFICATION_METRIC_HELP.linf, /mayor cambio/i)
    assert.match(CLASSIFICATION_METRIC_HELP.l2, /global/i)
})

test('las guías distinguen las unidades y agregaciones más ambiguas', () => {
    assert.match(DETECTION_METRIC_GUIDES.coco.items.map((item) => item.definition).join(' '), /puntos porcentuales/)
    assert.match(DETECTION_METRIC_GUIDES.tide.items.map((item) => item.term).join(' '), /dAP/)
    assert.match(CLASSIFICATION_METRIC_GUIDES.curves.items.map((item) => item.definition).join(' '), /micro/i)
    assert.match(CLASSIFICATION_METRIC_HELP.f1_macro, /clase/i)
})
