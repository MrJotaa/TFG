import assert from 'node:assert/strict'
import test from 'node:test'
import {
    analysisImagePosition,
    formatAnalysisProbability,
    formatSignedInteger,
    formatSignedPercentagePoints,
    normalizeClassificationPanel,
    normalizeDetectionPanel,
    resolveAdjacentAnalysisImage,
    resolveClassificationExplanation,
    resolveDetectionExplanation,
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


test('detection analysis keeps the same three workspace panels', () => {
    assert.equal(normalizeDetectionPanel('clean'), 'clean')
    assert.equal(normalizeDetectionPanel('attack'), 'attack')
    assert.equal(normalizeDetectionPanel('explanation'), 'explanation')
    assert.equal(normalizeDetectionPanel('unknown'), 'clean')
})

test('signed integer deltas describe detection count changes', () => {
    assert.equal(formatSignedInteger(-5), '-5')
    assert.equal(formatSignedInteger(2), '+2')
    assert.equal(formatSignedInteger(0), '0')
    assert.equal(formatSignedInteger(undefined), '—')
})

test('attack Eigen-CAM pair takes precedence in detection explainability', () => {
    const explanation = resolveDetectionExplanation({
        attackResult: {
            original_explanation_image_url: '/original-eigencam.png',
            adversarial_explanation_image_url: '/adversarial-eigencam.png',
            original_detections: [{ label: 'person', confidence: 0.91 }],
            adversarial_detections: [{ label: 'person', confidence: 0.61 }],
        },
        explainabilityResult: {
            explanation_image_url: '/standalone-eigencam.png',
            detections: [],
        },
    })

    assert.equal(explanation.mode, 'comparison')
    assert.equal(explanation.original.imageUrl, '/original-eigencam.png')
    assert.equal(explanation.adversarial.detections[0].confidence, 0.61)
})

test('standalone Eigen-CAM is available without an attack result', () => {
    assert.deepEqual(
        resolveDetectionExplanation({
            attackResult: null,
            explainabilityResult: {
                explanation_image_url: '/eigencam.png',
                detections: [{ label: 'car', confidence: 0.87 }],
            },
        }),
        {
            mode: 'single',
            original: {
                imageUrl: '/eigencam.png',
                detections: [{ label: 'car', confidence: 0.87 }],
            },
            adversarial: null,
        },
    )
})

test('adjacent image navigation is disabled during a controlled comparison and restored after it', () => {
    const images = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    assert.equal(resolveAdjacentAnalysisImage(images, 1, -1, true), null)
    assert.equal(resolveAdjacentAnalysisImage(images, 1, 1, true), null)
    assert.equal(resolveAdjacentAnalysisImage(images, 1, -1, false), images[0])
    assert.equal(resolveAdjacentAnalysisImage(images, 1, 1, false), images[2])
    assert.equal(resolveAdjacentAnalysisImage(images, 0, -1), null)
    assert.equal(resolveAdjacentAnalysisImage(images, 2, 1), null)
    assert.equal(resolveAdjacentAnalysisImage(images, -1, 1), null)
})
