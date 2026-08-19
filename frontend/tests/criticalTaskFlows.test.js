import assert from 'node:assert/strict'
import test from 'node:test'

import {
    getSupportedAttacksForTask,
    isAttackSupportedForTask,
} from '../src/utils/attackCatalog.js'
import { resolveExperimentTaskType } from '../src/utils/experimentTask.js'
import { activateModelInList } from '../src/utils/modelSelection.js'

const ATTACK_CATALOG = [
    { id: 'fgsm', supported_task_types: ['classification'] },
    { id: 'random_noise', supported_task_types: ['classification', 'detection'] },
    { id: 'pgd', supported_task_types: ['classification'] },
    { id: 'yolo_confidence', supported_task_types: ['detection'] },
]

test('classification exposes only classification-compatible attacks', () => {
    assert.deepEqual(
        getSupportedAttacksForTask(ATTACK_CATALOG, 'classification').map((attack) => attack.id),
        ['fgsm', 'random_noise', 'pgd'],
    )
    assert.equal(isAttackSupportedForTask(ATTACK_CATALOG, 'yolo_confidence', 'classification'), false)
    assert.equal(isAttackSupportedForTask(ATTACK_CATALOG, 'pgd', 'classification'), true)
})

test('detection exposes Random Noise and YOLO Confidence only', () => {
    assert.deepEqual(
        getSupportedAttacksForTask(ATTACK_CATALOG, 'detection').map((attack) => attack.id),
        ['random_noise', 'yolo_confidence'],
    )
    assert.equal(isAttackSupportedForTask(ATTACK_CATALOG, 'fgsm', 'detection'), false)
    assert.equal(isAttackSupportedForTask(ATTACK_CATALOG, 'yolo_confidence', 'detection'), true)
})

test('unknown attacks never become compatible by fallback', () => {
    assert.equal(isAttackSupportedForTask(ATTACK_CATALOG, 'unknown', 'classification'), false)
    assert.equal(isAttackSupportedForTask(ATTACK_CATALOG, 'unknown', 'detection'), false)
})

test('active model selects the evaluation task when there is no recovered result', () => {
    assert.equal(
        resolveExperimentTaskType({
            activeModel: { id: 'resnet18', task_type: 'classification' },
        }),
        'classification',
    )
    assert.equal(
        resolveExperimentTaskType({
            activeModel: { id: 'yolo26n', task_type: 'detection' },
        }),
        'detection',
    )
})

test('recovered evaluation task takes precedence over the currently active model', () => {
    assert.equal(
        resolveExperimentTaskType({
            activeModel: { id: 'resnet18', task_type: 'classification' },
            recoveredEvaluation: { taskType: 'detection' },
        }),
        'detection',
    )
    assert.equal(
        resolveExperimentTaskType({
            activeModel: { id: 'yolo26n', task_type: 'detection' },
            recoveredEvaluation: { taskType: 'classification' },
        }),
        'classification',
    )
})

test('task resolution defaults safely to classification before a model is selected', () => {
    assert.equal(resolveExperimentTaskType(), 'classification')
})


test('model selection leaves exactly one active and loaded model', () => {
    const models = [
        { id: 'resnet18', active: true, loaded: true },
        { id: 'yolo26n', active: false, loaded: false },
    ]

    const selected = activateModelInList(models, 'yolo26n')

    assert.deepEqual(selected, [
        { id: 'resnet18', active: false, loaded: false },
        { id: 'yolo26n', active: true, loaded: true },
    ])
    assert.equal(selected.filter((model) => model.active).length, 1)
    assert.equal(selected.filter((model) => model.loaded).length, 1)
})
