import assert from 'node:assert/strict'
import test from 'node:test'

import {
    DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
    filterAndSortDetectionObjects,
    getDetectionImageAggregateView,
    getDetectionObjectFilterOptions,
    paginateDetectionObjects,
} from '../src/utils/detectionObjectExplorer.js'

function object({
    id,
    image,
    categoryId,
    category,
    outcome = 'preserved',
    confidenceDrop = null,
    iouDrop = null,
    adversarialConfidence = null,
}) {
    return {
        object_id: id,
        image_id: image,
        outcome,
        confidence_drop: confidenceDrop,
        iou_drop: iouDrop,
        category: {
            category_internal_id: categoryId,
            category_name: category,
            model_class_name: category,
        },
        adversarial_observation: adversarialConfidence === null
            ? null
            : {
                confidence: adversarialConfidence,
                iou_with_ground_truth: 0.6,
            },
    }
}

const OBJECTS = [
    object({
        id: 'obj-1',
        image: 'image-b',
        categoryId: 'cat-dog',
        category: 'Perro',
        confidenceDrop: 0.1,
        iouDrop: 0.05,
        adversarialConfidence: 0.7,
    }),
    object({
        id: 'obj-2',
        image: 'image-a',
        categoryId: 'cat-car',
        category: 'Coche',
        outcome: 'suppressed',
    }),
    object({
        id: 'obj-3',
        image: 'image-a',
        categoryId: 'cat-dog',
        category: 'Perro',
        confidenceDrop: 0.4,
        iouDrop: -0.1,
        adversarialConfidence: 0.3,
    }),
]

test('construye filtros únicos y ordenados para imagen y clase', () => {
    const options = getDetectionObjectFilterOptions(OBJECTS)

    assert.deepEqual(options.images, ['image-a', 'image-b'])
    assert.deepEqual(options.categories, [
        { value: 'cat-car', label: 'Coche' },
        { value: 'cat-dog', label: 'Perro' },
    ])
})

test('combina búsqueda y filtros sobre el universo completo', () => {
    const filtered = filterAndSortDetectionObjects(OBJECTS, {
        query: 'per',
        outcome: 'preserved',
        imageId: 'image-a',
        categoryKey: 'cat-dog',
    })

    assert.deepEqual(filtered.map((item) => item.object_id), ['obj-3'])
})

test('ordena deltas numéricos de mayor a menor y deja nulos al final', () => {
    const sorted = filterAndSortDetectionObjects(OBJECTS, {
        sortBy: 'confidence_drop',
    })

    assert.deepEqual(sorted.map((item) => item.object_id), [
        'obj-3',
        'obj-1',
        'obj-2',
    ])
})

test('pagina después de filtrar y limita páginas fuera de rango', () => {
    const objects = Array.from({ length: 62 }, (_, index) => ({ object_id: index }))
    const middle = paginateDetectionObjects(objects, 2, 25)
    const last = paginateDetectionObjects(objects, 99, 25)

    assert.deepEqual(
        {
            page: middle.page,
            start: middle.start,
            end: middle.end,
            count: middle.items.length,
            totalPages: middle.totalPages,
        },
        { page: 2, start: 25, end: 50, count: 25, totalPages: 3 },
    )
    assert.equal(last.page, 3)
    assert.equal(last.items.length, 12)
})

test('resultados agregados por imagen se revelan de diez en diez', () => {
    const rows = Array.from({ length: 26 }, (_, index) => ({ key: `image-${index}` }))
    const first = getDetectionImageAggregateView(rows)
    const second = getDetectionImageAggregateView(rows, first.nextVisibleCount)
    const last = getDetectionImageAggregateView(rows, 30)

    assert.equal(DETECTION_IMAGE_AGGREGATE_BATCH_SIZE, 10)
    assert.deepEqual(
        { visible: first.visibleCount, total: first.totalCount, hasMore: first.hasMore, next: first.nextVisibleCount },
        { visible: 10, total: 26, hasMore: true, next: 20 },
    )
    assert.equal(second.items.length, 20)
    assert.equal(second.nextVisibleCount, 26)
    assert.equal(last.items.length, 26)
    assert.equal(last.hasMore, false)
})
