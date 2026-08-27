import test from 'node:test'
import assert from 'node:assert/strict'

import {
    baselineConsistency,
    classifyEvaluationComparison,
    getCompatibleEvaluationCandidates,
    getClassificationClassComparison,
    getComparisonImagePairs,
    getDetectionClassComparison,
    getDetectionCocoComparisonRows,
    getDetectionObjectComparisonPairs,
    getDetectionTideComparisonRows,
    getComparisonRetention,
    matchesComparisonImageFilter,
    pairComparisonMetrics,
    stableComparisonKey,
} from '../src/utils/evaluationComparison.js'

function historyItem(overrides = {}) {
    return {
        result_id: 'evaluation-a',
        task_type: 'classification',
        status: 'completed',
        dataset_id: 'dataset-1',
        dataset_name: 'Imagenette',
        dataset_revision: 4,
        model_id: 'resnet18',
        model_name: 'ResNet-18',
        attack_type: 'fgsm',
        parameters: { epsilon: 0.03 },
        supervised_configuration: null,
        ...overrides,
    }
}

function available(value) {
    return { available: true, value }
}

function classificationEvaluation({ clean = 0.8, robust = 0.5, drop = 0.3, asr = 0.4, f1 = 0.45 } = {}) {
    return {
        summary: {
            original_accuracy: clean,
            adversarial_accuracy: robust,
            accuracy_drop: drop,
        },
        advanced_metrics: {
            robustness: {
                clean_accuracy: available(clean),
                robust_accuracy: available(robust),
                accuracy_drop: available(drop),
                conditional_attack_success_rate: available(asr),
            },
            supervised: {
                aggregate: {
                    adversarial: { f1_macro: f1 },
                },
            },
        },
    }
}

test('stable comparison keys ignore object key order', () => {
    assert.equal(
        stableComparisonKey({ beta: 2, alpha: { y: 2, x: 1 } }),
        stableComparisonKey({ alpha: { x: 1, y: 2 }, beta: 2 }),
    )
})

test('same model with a different attack is a controlled attack comparison', () => {
    const comparison = classifyEvaluationComparison(
        historyItem(),
        historyItem({
            result_id: 'evaluation-b',
            attack_type: 'pgd',
            parameters: { epsilon: 0.03, alpha: 0.01, steps: 10 },
        }),
    )

    assert.equal(comparison.compatible, true)
    assert.equal(comparison.kind, 'attack')
})

test('different models require the same attack and parameters', () => {
    const comparison = classifyEvaluationComparison(
        historyItem(),
        historyItem({
            result_id: 'evaluation-b',
            model_id: 'swin-v2',
            model_name: 'Swin V2',
        }),
    )

    assert.equal(comparison.compatible, true)
    assert.equal(comparison.kind, 'model')
})

test('same model and attack with different parameters compares attack configuration', () => {
    const comparison = classifyEvaluationComparison(
        historyItem(),
        historyItem({
            result_id: 'evaluation-b',
            parameters: { epsilon: 0.08 },
        }),
    )

    assert.equal(comparison.compatible, true)
    assert.equal(comparison.kind, 'attack_configuration')
})

test('comparison rejects non-controlled or scientifically incompatible pairs', () => {
    const base = historyItem()
    const cases = [
        [historyItem(), 'same_result'],
        [historyItem({ result_id: 'b', task_type: 'detection' }), 'different_task'],
        [historyItem({ result_id: 'b', dataset_id: 'dataset-2' }), 'different_dataset'],
        [historyItem({ result_id: 'b', dataset_revision: 5 }), 'different_dataset_revision'],
        [
            historyItem({
                result_id: 'b',
                supervised_configuration: { confidence_threshold: 0.5 },
            }),
            'different_supervised_protocol',
        ],
        [historyItem({ result_id: 'b' }), 'identical_configuration'],
        [
            historyItem({
                result_id: 'b',
                model_id: 'swin-v2',
                attack_type: 'pgd',
                parameters: { epsilon: 0.08 },
            }),
            'multiple_variables',
        ],
    ]

    for (const [candidate, expectedCode] of cases) {
        const comparison = classifyEvaluationComparison(base, candidate)
        assert.equal(comparison.compatible, false)
        assert.equal(comparison.code, expectedCode)
    }
})

test('detection comparison requires the same supervised protocol', () => {
    const base = historyItem({
        task_type: 'detection',
        attack_type: 'random_noise',
        parameters: { magnitude: 0.1 },
        supervised_configuration: {
            confidence_threshold: 0.25,
            iou_threshold: 0.5,
            include_object_level_details: true,
        },
    })
    const candidate = historyItem({
        result_id: 'evaluation-b',
        task_type: 'detection',
        attack_type: 'random_noise',
        parameters: { magnitude: 0.2 },
        supervised_configuration: {
            confidence_threshold: 0.3,
            iou_threshold: 0.5,
            include_object_level_details: true,
        },
    })

    const comparison = classifyEvaluationComparison(base, candidate)
    assert.equal(comparison.compatible, false)
    assert.equal(comparison.code, 'different_supervised_protocol')
})

test('candidate filtering only keeps controlled comparisons', () => {
    const base = historyItem()
    const compatibleAttack = historyItem({
        result_id: 'attack-b',
        attack_type: 'pgd',
        parameters: { epsilon: 0.03, steps: 10 },
    })
    const compatibleModel = historyItem({
        result_id: 'model-b',
        model_id: 'swin-v2',
    })
    const incompatible = historyItem({
        result_id: 'mixed-b',
        model_id: 'swin-v2',
        attack_type: 'pgd',
    })

    const candidates = getCompatibleEvaluationCandidates(base, [
        base,
        compatibleAttack,
        compatibleModel,
        incompatible,
    ])

    assert.deepEqual(
        candidates.map(({ item, comparison }) => [item.result_id, comparison.kind]),
        [
            ['attack-b', 'attack'],
            ['model-b', 'model'],
        ],
    )
})

test('classification metric pairing unwraps available metrics and respects direction', () => {
    const metrics = pairComparisonMetrics(
        'classification',
        classificationEvaluation({ robust: 0.45, drop: 0.35, asr: 0.6 }),
        classificationEvaluation({ robust: 0.55, drop: 0.25, asr: 0.4 }),
    )
    const byKey = new Map(metrics.map((entry) => [entry.key, entry]))

    assert.equal(byKey.get('robust_accuracy').right, 0.55)
    assert.equal(byKey.get('robust_accuracy').advantage, 'right')
    assert.equal(byKey.get('accuracy_drop').advantage, 'right')
    assert.equal(byKey.get('conditional_asr').advantage, 'right')
})

test('baseline consistency compares the original reference for same-model analyses', () => {
    assert.deepEqual(
        baselineConsistency(
            'classification',
            classificationEvaluation({ clean: 0.8 }),
            classificationEvaluation({ clean: 0.8 }),
        ),
        { available: true, consistent: true, difference: 0 },
    )

    const mismatch = baselineConsistency(
        'classification',
        classificationEvaluation({ clean: 0.8 }),
        classificationEvaluation({ clean: 0.79 }),
    )
    assert.equal(mismatch.available, true)
    assert.equal(mismatch.consistent, false)
    assert.ok(mismatch.difference > 0)
})


test('classification detailed comparison aligns classes by label', () => {
    const left = {
        advanced_metrics: {
            supervised: {
                per_class: [{
                    class_reference: { index: 0, label: 'tench' },
                    support: 10,
                    original: { f1: available(0.9) },
                    adversarial: { f1: available(0.2) },
                }],
            },
        },
    }
    const right = {
        advanced_metrics: {
            supervised: {
                per_class: [{
                    class_reference: { index: 44, label: 'tench' },
                    support: 10,
                    original: { f1: available(0.88) },
                    adversarial: { f1: available(0.5) },
                }],
            },
        },
    }

    const [row] = getClassificationClassComparison(left, right, 'f1')
    assert.equal(row.label, 'tench')
    assert.equal(row.leftRobust, 0.2)
    assert.equal(row.rightRobust, 0.5)
    assert.ok(Math.abs(row.delta - 0.3) < 1e-12)
})

test('detection COCO detailed comparison exposes all robust metric rows', () => {
    const aggregate = {
        map_50_95: 0.3,
        ap_50: 0.5,
        ap_75: 0.25,
        ar_1: 0.1,
        ar_10: 0.2,
        ar_100: 0.4,
        ap_small: 0.11,
        ap_medium: 0.22,
        ap_large: 0.33,
        ar_small: 0.12,
        ar_medium: 0.23,
        ar_large: 0.34,
    }
    const left = { supervised: { adversarial: { aggregate } } }
    const right = { supervised: { adversarial: { aggregate: { ...aggregate, map_50_95: 0.4 } } } }
    const rows = getDetectionCocoComparisonRows(left, right)

    assert.equal(rows.length, 12)
    assert.equal(rows[0].key, 'map_50_95')
    assert.ok(Math.abs(rows[0].delta - 0.1) < 1e-12)
})

test('detection detailed class comparison aligns categories independently of model class ids', () => {
    const makeEvaluation = (modelClassId, ap, suppressed) => ({
        supervised: {
            original: {
                per_class: [{
                    reference: { category_internal_id: 'coco:1', category_name: 'person', model_class_id: modelClassId },
                    support: 20,
                }],
            },
            adversarial: {
                per_class: [{
                    reference: { category_internal_id: 'coco:1', category_name: 'person', model_class_id: modelClassId },
                    average_precision: ap,
                    average_recall: ap + 0.1,
                }],
            },
            object_analysis: {
                by_class: [{
                    category: { category_internal_id: 'coco:1', category_name: 'person' },
                    suppressed_count: suppressed,
                }],
            },
        },
    })

    const [row] = getDetectionClassComparison(
        makeEvaluation(0, 0.2, 8),
        makeEvaluation(1, 0.35, 5),
    )
    assert.equal(row.label, 'person')
    assert.equal(row.leftSuppressed, 8)
    assert.equal(row.rightSuppressed, 5)
    assert.ok(Math.abs(row.delta - 0.15) < 1e-12)
})

test('image and object comparisons only keep entities present in both evaluations', () => {
    const imagePairs = getComparisonImagePairs(
        { results: [{ image_id: '1' }, { image_id: '2' }] },
        { results: [{ image_id: '2' }, { image_id: '3' }] },
    )
    assert.deepEqual(imagePairs.map(({ left }) => left.image_id), ['2'])

    const objectPairs = getDetectionObjectComparisonPairs(
        { supervised: { object_analysis: { objects: [{ object_id: 'a' }, { object_id: 'b' }] } } },
        { supervised: { object_analysis: { objects: [{ object_id: 'b' }] } } },
    )
    assert.deepEqual(objectPairs.map(({ left }) => left.object_id), ['b'])
})

test('TIDE comparison pairs adversarial error impact and counts by error type', () => {
    const makeEvaluation = (deltaAp, count) => ({
        supervised: {
            tide_analysis: {
                adversarial: { main_errors: [{ error_type: 'missed', delta_ap: deltaAp, count }] },
            },
        },
    })
    const [row] = getDetectionTideComparisonRows(
        makeEvaluation(12.5, 20),
        makeEvaluation(18.0, 30),
    )
    assert.equal(row.key, 'missed')
    assert.equal(row.leftImpact, 12.5)
    assert.equal(row.rightCount, 30)
})


test('retention profile relates robust performance to each evaluation baseline', () => {
    assert.deepEqual(
        getComparisonRetention('classification', classificationEvaluation({ clean: 0.8, robust: 0.4 })),
        { original: 0.8, robust: 0.4, retention: 0.5 },
    )

    const detection = {
        supervised: {
            original: { aggregate: { map_50_95: 0.4 } },
            adversarial: { aggregate: { map_50_95: 0.1 } },
        },
    }
    assert.deepEqual(
        getComparisonRetention('detection', detection),
        { original: 0.4, robust: 0.1, retention: 0.25 },
    )
})

test('classification image filters distinguish which evaluation keeps the correct label', () => {
    const pair = {
        left: { ground_truth_label: 'tench', adversarial_label: 'tench', adversarial_correct: true },
        right: { ground_truth_label: 'tench', adversarial_label: 'crab', adversarial_correct: false },
    }
    assert.equal(matchesComparisonImageFilter('classification', pair, 'left_better'), true)
    assert.equal(matchesComparisonImageFilter('classification', pair, 'right_better'), false)
    assert.equal(matchesComparisonImageFilter('classification', pair, 'both_fail'), false)
})

test('detection image filters compare adversarial detection counts without inferring quality', () => {
    const pair = {
        left: { adversarial_detection_count: 5 },
        right: { adversarial_detection_count: 2 },
    }
    assert.equal(matchesComparisonImageFilter('detection', pair, 'left_more'), true)
    assert.equal(matchesComparisonImageFilter('detection', pair, 'right_more'), false)
    assert.equal(matchesComparisonImageFilter('detection', pair, 'different'), true)
    assert.equal(matchesComparisonImageFilter('detection', pair, 'equal'), false)
})
