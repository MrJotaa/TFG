import test from 'node:test'
import assert from 'node:assert/strict'

import {
    evaluationExpirationLabel,
    formatRemainingTime,
    historySummary,
    historyTotalLabel,
    isNearExpiration,
} from '../src/utils/evaluationHistory.js'

test('remaining time uses compact human-readable units', () => {
    const now = Date.parse('2026-08-03T10:00:00Z')
    assert.equal(formatRemainingTime('2026-08-03T10:45:00Z', now), '45 min')
    assert.equal(formatRemainingTime('2026-08-03T13:00:00Z', now), '3 h')
    assert.equal(formatRemainingTime('2026-08-06T10:00:00Z', now), '3 días')
})

test('near expiration uses ten percent with a two-hour ceiling', () => {
    const now = Date.parse('2026-08-03T10:00:00Z')
    assert.equal(
        isNearExpiration(
            {
                expires_at: '2026-08-03T11:00:00Z',
                retention_seconds: 604800,
            },
            now,
        ),
        true,
    )
    assert.equal(
        isNearExpiration(
            {
                expires_at: '2026-08-03T13:00:00Z',
                retention_seconds: 604800,
            },
            now,
        ),
        false,
    )
})

test('history summaries distinguish classification and detection', () => {
    assert.equal(
        historySummary({
            task_type: 'classification',
            summary: { original_accuracy: 0.8, adversarial_accuracy: 0.55 },
        }),
        'Accuracy 80.0% → 55.0%',
    )
    assert.equal(
        historySummary({
            task_type: 'detection',
            summary: {
                total_original_detections: 42,
                total_adversarial_detections: 31,
            },
        }),
        '42 → 31 detecciones',
    )
})


test('evaluation expiration labels distinguish running, preserved and expiring results', () => {
    const now = Date.parse('2026-08-03T10:00:00Z')
    assert.equal(evaluationExpirationLabel({ status: 'running' }, now), 'Al finalizar')
    assert.equal(evaluationExpirationLabel({ preserved: true }, now), 'Indefinida')
    assert.equal(
        evaluationExpirationLabel({ expires_at: '2026-08-03T11:30:00Z' }, now),
        '2 h',
    )
})

test('history total labels use the correct singular and plural nouns', () => {
    assert.equal(historyTotalLabel('evaluations', 1), '1 evaluación vigente')
    assert.equal(historyTotalLabel('evaluations', 2), '2 evaluaciones vigentes')
    assert.equal(historyTotalLabel('individual', 1), '1 análisis guardado')
    assert.equal(historyTotalLabel('individual', 3), '3 análisis guardados')
})
