import test from 'node:test'
import assert from 'node:assert/strict'

import {
    classifyIndividualAnalysisComparison,
    getCompatibleIndividualAnalysisCandidates,
    individualAnalysisOutcome,
    individualAnalysisSummary,
} from '../src/utils/individualAnalysisHistory.js'

function historyItem(overrides = {}) {
    return {
        analysis_id: 'analysis-a',
        task_type: 'classification',
        dataset_id: 'dataset-1',
        dataset_name: 'Imagenette',
        dataset_revision: 3,
        image_id: 'image-1',
        image_filename: 'tench.JPEG',
        model_id: 'resnet18',
        model_name: 'ResNet-18',
        model_revision: 'resnet18@1',
        attack_type: 'fgsm',
        attack_revision: 'fgsm@1',
        parameters: { epsilon: 0.03 },
        summary: {
            original_label: 'tench',
            adversarial_label: 'goldfish',
            attack_success: true,
        },
        explanation_available: true,
        pipeline_version: 'classification-individual-analysis-v1',
        ...overrides,
    }
}

test('individual history presents task-specific compact summaries', () => {
    assert.equal(individualAnalysisSummary(historyItem()), 'tench → goldfish')
    assert.equal(individualAnalysisOutcome(historyItem()), 'Ataque exitoso')

    const detection = historyItem({
        task_type: 'detection',
        summary: {
            original_detection_count: 8,
            adversarial_detection_count: 3,
            attack_success: false,
        },
    })
    assert.equal(individualAnalysisSummary(detection), '8 → 3 detecciones')
    assert.equal(individualAnalysisOutcome(detection), 'Criterio histórico no cumplido')
})

test('individual comparison requires the same dataset revision and exact image', () => {
    const base = historyItem()

    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'b',
            dataset_revision: 4,
        })).code,
        'different_dataset_revision',
    )
    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'b',
            image_id: 'image-2',
        })).code,
        'different_image',
    )
})


test('individual comparison rejects hidden implementation revision changes', () => {
    const base = historyItem()

    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'model-revision-b',
            model_revision: 'resnet18@2',
        })).code,
        'different_model_revision',
    )
    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'attack-revision-b',
            attack_revision: 'fgsm@2',
            parameters: { epsilon: 0.08 },
        })).code,
        'different_attack_revision',
    )
    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'pipeline-b',
            pipeline_version: 'classification-individual-analysis-v2',
            attack_type: 'pgd',
        })).code,
        'different_pipeline_version',
    )
})

test('individual history classifies controlled model, attack and parameter comparisons', () => {
    const base = historyItem()

    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'model-b',
            model_id: 'swin-v2',
            model_name: 'Swin V2',
            model_revision: 'swin-v2@1',
        })).kind,
        'model',
    )
    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'attack-b',
            attack_type: 'pgd',
            attack_revision: 'pgd@1',
            parameters: { epsilon: 0.03, steps: 10 },
        })).kind,
        'attack',
    )
    assert.equal(
        classifyIndividualAnalysisComparison(base, historyItem({
            analysis_id: 'config-b',
            parameters: { epsilon: 0.08 },
        })).kind,
        'attack_configuration',
    )
})

test('candidate preparation rejects identical or multi-variable individual analyses', () => {
    const base = historyItem()
    const compatible = historyItem({
        analysis_id: 'attack-b',
        attack_type: 'pgd',
        attack_revision: 'pgd@1',
        parameters: { epsilon: 0.03, steps: 10 },
    })
    const mixed = historyItem({
        analysis_id: 'mixed-b',
        model_id: 'swin-v2',
        attack_type: 'pgd',
    })

    const candidates = getCompatibleIndividualAnalysisCandidates(base, [
        base,
        compatible,
        mixed,
    ])

    assert.deepEqual(
        candidates.map(({ item, comparison }) => [item.analysis_id, comparison.kind]),
        [['attack-b', 'attack']],
    )
})


test('missing individual artifacts are excluded from saved comparison candidates', () => {
    const base = historyItem()
    const other = historyItem({ analysis_id: 'other', attack_type: 'pgd', artifact_available: true })
    assert.equal(getCompatibleIndividualAnalysisCandidates(base, [{ ...other, artifact_available: false }]).length, 0)
    assert.equal(getCompatibleIndividualAnalysisCandidates(base, [other]).length, 1)
    assert.equal(getCompatibleIndividualAnalysisCandidates({ ...base, artifact_available: false }, [other]).length, 0)
})
