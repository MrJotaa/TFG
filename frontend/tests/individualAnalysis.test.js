import assert from 'node:assert/strict'
import test from 'node:test'
import {
    analysisImagePosition,
    formatAnalysisProbability,
    formatSignedPercentagePoints,
    normalizeClassificationPanel,
    resolveClassificationExplanation,
} from '../src/utils/individualAnalysis.js'

test('classification analysis only accepts known panels', () => {
    assert.equal(normalizeClassificationPanel('attack'), 'attack')
    assert.equal(normalizeClassificationPanel('explanation'), 'explanation')
    assert.equal(normalizeClassificationPanel('unknown'), 'clean')
})

test('probabilities are rendered as percentages', () => {
    assert.equal(formatAnalysisProbability(0.9817), '98.17%')
    assert.equal(formatAnalysisProbability('0.5', 1), '50.0%')
    assert.equal(formatAnalysisProbability(undefined), '—')
})

test('confidence differences are rendered as signed percentage points', () => {
    assert.equal(formatSignedPercentagePoints(-0.784), '-78.40 pp')
    assert.equal(formatSignedPercentagePoints(0.125), '+12.50 pp')
    assert.equal(formatSignedPercentagePoints('invalid'), '—')
})

test('attack Grad-CAM pair takes precedence in explainability', () => {
    const result = resolveClassificationExplanation({
        attackResult: {
            original_gradcam_image_url: '/original-cam.png',
            adversarial_gradcam_image_url: '/adversarial-cam.png',
            original_prediction: { prediction: 'church', confidence: 0.98 },
            adversarial_prediction: { prediction: 'castle', confidence: 0.2 },
        },
        explainabilityResult: {
            gradcam_image_url: '/standalone-cam.png',
            prediction: 'church',
            confidence: 0.98,
        },
    })

    assert.equal(result.mode, 'comparison')
    assert.equal(result.original.imageUrl, '/original-cam.png')
    assert.equal(result.adversarial.prediction, 'castle')
})

test('standalone Grad-CAM is used when no attack comparison exists', () => {
    assert.deepEqual(
        resolveClassificationExplanation({
            attackResult: null,
            explainabilityResult: {
                gradcam_image_url: '/cam.png',
                prediction: 'church',
                confidence: 0.93,
            },
        }),
        {
            mode: 'single',
            original: {
                imageUrl: '/cam.png',
                prediction: 'church',
                confidence: 0.93,
            },
            adversarial: null,
        },
    )
})

test('image position is compact and safe for missing selections', () => {
    assert.equal(analysisImagePosition(47, 100), '48 / 100')
    assert.equal(analysisImagePosition(-1, 100), '— / —')
})
