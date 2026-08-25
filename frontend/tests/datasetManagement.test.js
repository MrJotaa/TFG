import test from 'node:test'
import assert from 'node:assert/strict'

import {
    datasetSupportsAnnotations,
    filterClassificationImages,
    getClassificationLabelStats,
    getDatasetSidebarMeta,
    getDatasetTaskLabel,
    getSelectedFilesSummary,
} from '../src/utils/datasetManagement.js'

test('dataset task labels distinguish classification, detection and legacy datasets', () => {
    assert.equal(getDatasetTaskLabel('classification'), 'Clasificación')
    assert.equal(getDatasetTaskLabel('detection'), 'Detección')
    assert.equal(getDatasetTaskLabel(null), 'Tipo sin definir')
})

test('COCO tools are exposed only for detection datasets', () => {
    assert.equal(datasetSupportsAnnotations({ task_type: 'detection' }), true)
    assert.equal(datasetSupportsAnnotations({ task_type: 'classification' }), false)
    assert.equal(datasetSupportsAnnotations({ task_type: null }), false)
})

test('classification label stats count labeled and unlabeled images', () => {
    const stats = getClassificationLabelStats([
        { label: 'cat' },
        { label: '  dog  ' },
        { label: '' },
        { label: null },
    ])

    assert.deepEqual(stats, {
        total: 4,
        labeled: 2,
        unlabeled: 2,
        percentage: 50,
    })
})

test('classification image filters preserve the expected samples', () => {
    const images = [
        { id: '1', label: 'cat' },
        { id: '2', label: null },
        { id: '3', label: 'dog' },
    ]

    assert.equal(filterClassificationImages(images, 'all').length, 3)
    assert.deepEqual(
        filterClassificationImages(images, 'labeled').map((image) => image.id),
        ['1', '3'],
    )
    assert.deepEqual(
        filterClassificationImages(images, 'unlabeled').map((image) => image.id),
        ['2'],
    )
})

test('bulk image selection summarizes many files without listing all of them', () => {
    const summary = getSelectedFilesSummary([
        { name: '1.png', size: 1024 * 1024 },
        { name: '2.png', size: 2 * 1024 * 1024 },
        { name: '3.png', size: 1024 },
        { name: '4.png', size: 2048 },
    ])

    assert.equal(summary.count, 4)
    assert.equal(summary.previewNames.length, 3)
    assert.equal(summary.remainingCount, 1)
    assert.ok(summary.totalMegabytes > 3)
})

test('dataset sidebar metadata communicates type and image count', () => {
    assert.equal(
        getDatasetSidebarMeta({ task_type: 'detection', image_count: 100 }),
        'Detección · 100 imágenes',
    )
    assert.equal(
        getDatasetSidebarMeta({ task_type: null, image_count: 1 }),
        'Tipo sin definir · 1 imagen',
    )
})
