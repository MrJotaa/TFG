import test from 'node:test'
import assert from 'node:assert/strict'

import {
    formatRemainingTime,
    historySummary,
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
