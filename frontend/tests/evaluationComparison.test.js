import test from 'node:test'
import assert from 'node:assert/strict'

import {
    baselineConsistency,
    classifyEvaluationComparison,
    getCompatibleEvaluationCandidates,
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
