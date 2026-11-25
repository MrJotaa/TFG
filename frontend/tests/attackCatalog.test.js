import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
    attackParametersKey,
    buildAttackParameters,
    getAttackParameterError,
    getSupportedAttacksForTask,
    isAttackSupportedForTask,
    resolveCompatibleAttackSelection,
} from '../src/utils/attackCatalog.js'
import {
    formatAttackParameters,
    getAttackName,
} from '../src/utils/attackFormatting.js'

const catalog = [
    {
        id: 'fgsm',
        name: 'Fast Gradient Sign Method',
        supported_task_types: ['classification'],
        parameters: [
            {
                key: 'epsilon',
                summary_label: 'ε',
                default: 0.03,
            },
        ],
    },
    {
        id: 'random_noise',
        name: 'Random Noise Baseline',
        supported_task_types: ['classification', 'detection'],
        parameters: [
            {
                key: 'epsilon',
                summary_label: 'ε',
                default: 0.03,
            },
            {
                key: 'seed',
                summary_label: 'seed',
                default: 0,
            },
        ],
    },
    {
        id: 'carlini_wagner',
        name: 'Carlini & Wagner',
        supported_task_types: ['classification'],
        parameters: [
            {
                key: 'norm',
                summary_label: 'norm',
                default: 'l2',
                choices: [
                    { value: 'l2', label: 'L2' },
                    { value: 'linf', label: 'L∞' },
                ],
            },
        ],
    },
]

test('catalog compatibility is driven by backend metadata', () => {
    assert.equal(isAttackSupportedForTask(catalog, 'fgsm', 'classification'), true)
    assert.equal(isAttackSupportedForTask(catalog, 'fgsm', 'detection'), false)
    assert.deepEqual(
        getSupportedAttacksForTask(catalog, 'detection').map((attack) => attack.id),
        ['random_noise'],
    )
})

test('attack changes retain shared values and discard irrelevant parameters', () => {
    const parameters = buildAttackParameters(catalog, 'random_noise', {
        epsilon: 0.08,
        alpha: 0.01,
        steps: 10,
    })

    assert.deepEqual(parameters, {
        epsilon: 0.08,
        seed: 0,
    })
})

test('parameter identity is stable regardless of object insertion order', () => {
    assert.equal(
        attackParametersKey({ epsilon: 0.03, seed: 7 }),
        attackParametersKey({ seed: 7, epsilon: 0.03 }),
    )
})

test('attack summaries use catalog names and parameter metadata', () => {
    assert.equal(getAttackName('fgsm', catalog), 'Fast Gradient Sign Method')
    assert.equal(
        formatAttackParameters('random_noise', { epsilon: 0.05, seed: 9 }, catalog),
        'ε=0.05 · seed=9',
    )
})


test('select parameters use catalog choice labels in summaries', () => {
    assert.equal(
        formatAttackParameters('carlini_wagner', { norm: 'linf' }, catalog),
        'norm=L∞',
    )
})

test('dynamic parameter builder supports ART numeric select and boolean controls', () => {
    const artCatalog = [
        {
            id: 'zoo',
            name: 'Zeroth-Order Optimization (ZOO)',
            supported_task_types: ['classification'],
            parameters: [
                {
                    key: 'confidence',
                    value_type: 'float',
                    control: 'number',
                    summary_label: 'κ',
                    default: 0,
                },
                {
                    key: 'max_iter',
                    value_type: 'integer',
                    control: 'number',
                    summary_label: 'iter',
                    default: 10,
                },
                {
                    key: 'use_resize',
                    value_type: 'boolean',
                    control: 'checkbox',
                    summary_label: 'resize',
                    default: true,
                },
            ],
        },
        {
            id: 'hop_skip_jump',
            name: 'HopSkipJump',
            supported_task_types: ['classification'],
            parameters: [
                {
                    key: 'norm',
                    value_type: 'string',
                    control: 'select',
                    summary_label: 'norm',
                    default: 'l2',
                    choices: [
                        { value: 'l2', label: 'L2' },
                        { value: 'linf', label: 'L∞' },
                    ],
                },
            ],
        },
    ]

    assert.deepEqual(buildAttackParameters(artCatalog, 'zoo'), {
        confidence: 0,
        max_iter: 10,
        use_resize: true,
    })
    assert.deepEqual(
        buildAttackParameters(artCatalog, 'hop_skip_jump', {
            norm: 'linf',
            max_iter: 999,
            use_resize: false,
        }),
        { norm: 'linf' },
    )
    assert.equal(
        formatAttackParameters('hop_skip_jump', { norm: 'linf' }, artCatalog),
        'norm=L∞',
    )
})

const detectionCatalog = [
    ...catalog,
    {
        id: 'yolo_confidence',
        name: 'Supresión de confianza',
        supported_task_types: ['detection'],
        parameters: [{ key: 'steps', default: 10 }],
    },
    {
        id: 'detection_localization_pgd',
        name: 'PGD de localización',
        supported_task_types: ['detection'],
        supported_model_ids: ['rf_detr_nano'],
        parameters: [
            { key: 'epsilon', default: 0.03 },
            { key: 'steps', default: 20 },
        ],
    },
]

test('the catalog filters attacks by task and model without restricting legacy entries', () => {
    assert.deepEqual(
        getSupportedAttacksForTask(detectionCatalog, 'detection', 'yolo26n')
            .map((attack) => attack.id),
        ['random_noise', 'yolo_confidence'],
    )
    assert.deepEqual(
        getSupportedAttacksForTask(detectionCatalog, 'detection', 'rf_detr_nano')
            .map((attack) => attack.id),
        ['random_noise', 'yolo_confidence', 'detection_localization_pgd'],
    )
    assert.deepEqual(
        getSupportedAttacksForTask(detectionCatalog, 'classification', 'resnet18')
            .map((attack) => attack.id),
        ['fgsm', 'random_noise', 'carlini_wagner'],
    )
    assert.equal(
        isAttackSupportedForTask(
            detectionCatalog, 'detection_localization_pgd', 'detection', 'yolo26n',
        ),
        false,
    )
    assert.equal(
        isAttackSupportedForTask(
            detectionCatalog, 'detection_localization_pgd', 'detection', 'rf_detr_nano',
        ),
        true,
    )
    // Legacy consumers that ask only for a task retain their existing semantics.
    assert.equal(
        isAttackSupportedForTask(detectionCatalog, 'detection_localization_pgd', 'detection'),
        true,
    )
})

test('changing from classification to detection falls back to noise and drops hidden parameters', () => {
    assert.deepEqual(
        resolveCompatibleAttackSelection(
            detectionCatalog,
            { id: 'yolo26n', task_type: 'detection' },
            'carlini_wagner',
            { norm: 'linf', epsilon: 0.08, max_iter: 100 },
        ),
        { attackId: 'random_noise', parameters: { epsilon: 0.08, seed: 0 } },
    )
})

test('switching between detectors rejects RF-DETR-only attacks but preserves shared ones', () => {
    const yolo = { id: 'yolo26n', task_type: 'detection' }
    const rfDetr = { id: 'rf_detr_nano', task_type: 'detection' }

    assert.deepEqual(
        resolveCompatibleAttackSelection(
            detectionCatalog, rfDetr, 'detection_localization_pgd',
            { epsilon: 0.02, steps: 35 },
        ),
        {
            attackId: 'detection_localization_pgd',
            parameters: { epsilon: 0.02, steps: 35 },
        },
    )
    assert.deepEqual(
        resolveCompatibleAttackSelection(
            detectionCatalog, yolo, 'detection_localization_pgd',
            { epsilon: 0.02, steps: 35 },
        ),
        { attackId: 'random_noise', parameters: { epsilon: 0.02, seed: 0 } },
    )
    assert.deepEqual(
        resolveCompatibleAttackSelection(detectionCatalog, yolo, 'random_noise', {
            epsilon: 0.04, seed: 11,
        }),
        { attackId: 'random_noise', parameters: { epsilon: 0.04, seed: 11 } },
    )
})

test('fallback handles an unavailable baseline, no compatible attacks, and late catalog loading', () => {
    const model = { id: 'rf_detr_nano', task_type: 'detection' }
    const onlyDetection = [
        { id: 'detector_attack', supported_task_types: ['detection'], parameters: [] },
    ]
    assert.deepEqual(
        resolveCompatibleAttackSelection(onlyDetection, model, 'fgsm', { epsilon: 0.1 }),
        { attackId: 'detector_attack', parameters: {} },
    )
    assert.deepEqual(
        resolveCompatibleAttackSelection(catalog, model, 'fgsm', { epsilon: 0.1 }),
        { attackId: 'random_noise', parameters: { epsilon: 0.1, seed: 0 } },
    )
    assert.deepEqual(
        resolveCompatibleAttackSelection([catalog[0]], model, 'fgsm', { epsilon: 0.1 }),
        { attackId: null, parameters: {} },
    )
    assert.deepEqual(
        resolveCompatibleAttackSelection([], model, 'fgsm', { epsilon: 0.1 }),
        { attackId: 'fgsm', parameters: { epsilon: 0.1 } },
    )
    assert.deepEqual(
        resolveCompatibleAttackSelection(detectionCatalog, null, 'fgsm', { epsilon: 0.1 }),
        { attackId: 'fgsm', parameters: { epsilon: 0.1 } },
    )
})

test('historical attack ids and parameters remain readable even if not offered to the active model', () => {
    assert.equal(
        isAttackSupportedForTask(
            detectionCatalog, 'detection_localization_pgd', 'detection', 'yolo26n',
        ),
        false,
    )
    assert.equal(getAttackName('detection_localization_pgd', detectionCatalog), 'PGD de localización')
    assert.deepEqual(
        buildAttackParameters(detectionCatalog, 'detection_localization_pgd', {
            epsilon: 0.05, steps: 50,
        }),
        { epsilon: 0.05, steps: 50 },
    )
})

const hopSkipJumpCatalog = [{
    id: 'hop_skip_jump',
    parameters: [
        { key: 'init_eval', default: 100, maximum: 10000, label: 'Evaluaciones iniciales' },
        { key: 'max_eval', default: 10000, maximum: 10000, label: 'Evaluaciones máximas por iteración' },
    ],
}]

test('HopSkipJump validates the relation between initial and maximum evaluations', () => {
    const validate = (parameters) => getAttackParameterError(
        hopSkipJumpCatalog, 'hop_skip_jump', parameters,
    )
    assert.equal(validate({}), '')
    assert.equal(validate({ init_eval: 100, max_eval: 100 }), '')
    assert.equal(validate({ init_eval: 50, max_eval: 100 }), '')
    assert.match(validate({ init_eval: 100, max_eval: 50 }), /menores o iguales/)
    assert.match(validate({ max_eval: 50 }), /menores o iguales/)
    assert.equal(validate({ init_eval: 1000 }), '')
})

test('HopSkipJump blocks empty or invalid evaluation counts', () => {
    for (const value of [NaN, null, '', 0, -1, 1.5, Infinity]) {
        for (const key of ['init_eval', 'max_eval']) {
            assert.match(getAttackParameterError(hopSkipJumpCatalog, 'hop_skip_jump', {
                [key]: value,
            }), /entero positivo/)
        }
    }
    assert.match(getAttackParameterError(hopSkipJumpCatalog, 'hop_skip_jump', {
        max_eval: 10001,
    }), /10000/)
})

test('HopSkipJump cross-parameter constraints do not affect other attacks', () => {
    assert.equal(getAttackParameterError(catalog, 'fgsm', {
        init_eval: 100, max_eval: 50,
    }), '')
})
