import assert from 'node:assert/strict'
import test from 'node:test'
import {
    buildExplorerSources,
    clampExplorerZoom,
    resolveCleanDetectionResult,
    resolveExplorerPair,
    resolveOverlaySplit,
} from '../src/utils/adversarialExplorer.js'

test('clampExplorerZoom keeps zoom inside the supported interval', () => {
    assert.equal(clampExplorerZoom(0), 1)
    assert.equal(clampExplorerZoom(2.5), 2.5)
    assert.equal(clampExplorerZoom(8), 4)
    assert.equal(clampExplorerZoom('invalid'), 1)
})

test('overlay split keeps original on the left and adversarial on the right', () => {
    assert.deepEqual(resolveOverlaySplit(48), {
        originalPercent: 48,
        adversarialPercent: 52,
        originalClipPath: 'inset(0 52% 0 0)',
    })

    assert.deepEqual(resolveOverlaySplit(0), {
        originalPercent: 0,
        adversarialPercent: 100,
        originalClipPath: 'inset(0 100% 0 0)',
    })

    assert.deepEqual(resolveOverlaySplit(100), {
        originalPercent: 100,
        adversarialPercent: 0,
        originalClipPath: 'inset(0 0% 0 0)',
    })
})

test('classification sources preserve real images and Grad-CAM layers', () => {
    const sources = buildExplorerSources({
        taskType: 'classification',
        selectedImage: { url: '/datasets/1/images/clean' },
        attackResult: {
            adversarial_image_url: '/results/1/adversarial.png',
            perturbation_image_url: '/results/1/perturbation.png',
            original_gradcam_image_url: '/results/1/original-gradcam.png',
            adversarial_gradcam_image_url: '/results/1/adversarial-gradcam.png',
        },
    })

    assert.equal(sources.supportsAnnotations, false)
    assert.equal(sources.supportsExplanations, true)
    assert.deepEqual(
        resolveExplorerPair({
            sources,
            layer: 'image',
            showAnnotations: false,
        }),
        {
            original: '/datasets/1/images/clean',
            adversarial: '/results/1/adversarial.png',
        },
    )
    assert.deepEqual(
        resolveExplorerPair({
            sources,
            layer: 'explanation',
            showAnnotations: false,
        }),
        {
            original: '/results/1/original-gradcam.png',
            adversarial: '/results/1/adversarial-gradcam.png',
        },
    )
})

test('detection sources switch boxes without replacing adversarial pixels', () => {
    const sources = buildExplorerSources({
        taskType: 'detection',
        selectedImage: { url: '/datasets/2/images/clean' },
        attackResult: {
            adversarial_image_url: '/results/2/adversarial.png',
            original_annotated_image_url: '/results/2/original-boxes.png',
            adversarial_annotated_image_url: '/results/2/adversarial-boxes.png',
            perturbation_image_url: '/results/2/perturbation.png',
            original_explanation_image_url: '/results/2/original-cam.png',
            adversarial_explanation_image_url: '/results/2/adversarial-cam.png',
        },
    })

    assert.equal(sources.supportsAnnotations, true)
    assert.deepEqual(
        resolveExplorerPair({
            sources,
            layer: 'image',
            showAnnotations: true,
        }),
        {
            original: '/results/2/original-boxes.png',
            adversarial: '/results/2/adversarial-boxes.png',
        },
    )
    assert.deepEqual(
        resolveExplorerPair({
            sources,
            layer: 'image',
            showAnnotations: false,
        }),
        {
            original: '/datasets/2/images/clean',
            adversarial: '/results/2/adversarial.png',
        },
    )
})

test('missing attack result does not create partial explorer sources', () => {
    assert.equal(
        buildExplorerSources({
            taskType: 'classification',
            selectedImage: { url: '/image.png' },
            attackResult: null,
        }),
        null,
    )
})

test('clean detection reuses the original side of an attack result', () => {
    const attackResult = {
        original_annotated_image_url: '/results/3/original-boxes.png',
        original_detections: [{ label: 'person', confidence: 0.91 }],
    }

    assert.deepEqual(
        resolveCleanDetectionResult({ detectionResult: null, attackResult }),
        {
            annotated_image_url: '/results/3/original-boxes.png',
            detections: attackResult.original_detections,
        },
    )
})

test('explicit clean detection takes precedence over the attack reference', () => {
    const detectionResult = {
        annotated_image_url: '/detection/clean.png',
        detections: [{ label: 'car', confidence: 0.87 }],
    }

    assert.equal(
        resolveCleanDetectionResult({
            detectionResult,
            attackResult: {
                original_annotated_image_url: '/results/old.png',
                original_detections: [],
            },
        }),
        detectionResult,
    )
})
