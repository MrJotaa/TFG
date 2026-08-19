import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
    attackParametersKey,
    buildAttackParameters,
    getSupportedAttacksForTask,
    isAttackSupportedForTask,
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
