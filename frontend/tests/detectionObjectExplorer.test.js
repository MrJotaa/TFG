import assert from 'node:assert/strict'
import test from 'node:test'

import {
    DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
    filterAndSortDetectionObjects,
    getDetectionImageAggregateView,
    getDetectionObjectImagePreview,
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


test('la inspección usa el ID de la imagen, sin depender del orden de resultados', () => {
    const evaluation = {
        dataset_id: 'dataset-1',
        results: [
            { image_id: 'b', filename: 'b.jpg', adversarial_image_url: '/results/b.png' },
            { image_id: 'a', filename: 'a.jpg', adversarial_image_url: '/results/a.png' },
        ],
        supervised: { object_analysis: {
            by_image: [{ image_id: 'a', preserved_count: 1 }],
            objects: [
                { image_id: 'b', object_id: 'otro' },
                {
                    image_id: 'a', object_id: 'obj-1',
                    category: { category_name: 'Coche' },
                    ground_truth_box: { x1: 5, y1: 10, x2: 50, y2: 60 },
                    original_observation: {
                        confidence: 0.8, detection_box: { x1: 4, y1: 9, x2: 49, y2: 59 },
                    },
                    adversarial_observation: null,
                },
            ],
        } },
    }
    const preview = getDetectionObjectImagePreview(evaluation, 'a')
    assert.equal(preview.filename, 'a.jpg')
    assert.equal(preview.adversarialImagePath, '/results/a.png')
    assert.equal(preview.originalImagePath, '/datasets/dataset-1/images/a')
    assert.equal(preview.aggregate.preserved_count, 1)
    assert.deepEqual(preview.objects.map((item) => item.object_id), ['obj-1'])
    assert.equal(preview.originalDetections.length, 2)
    assert.equal(preview.originalDetections[0].tone, 'reference')
    assert.equal(preview.originalDetections[1].tone, 'original')
    assert.equal(preview.adversarialDetections.length, 1)
    assert.equal(preview.adversarialDetections[0].tone, 'reference')
})

test('si faltan artefactos históricos no se sustituyen por imágenes de otra evaluación', () => {
    const preview = getDetectionObjectImagePreview({
        dataset_id: 'dataset/temporal',
        results: [],
        supervised: { object_analysis: { by_image: [], objects: [
            { image_id: 'imagen 1', ground_truth_box: { x1: 1, y1: 1, x2: 1, y2: 2 } },
        ] } },
    }, 'imagen 1')
    assert.equal(preview.filename, null)
    assert.equal(preview.adversarialImagePath, null)
    assert.equal(preview.originalImagePath, '/datasets/dataset%2Ftemporal/images/imagen%201')
    assert.deepEqual(preview.originalDetections, [])
    assert.deepEqual(preview.adversarialDetections, [])
})

const BOX = { x1: 10, y1: 12, x2: 70, y2: 82 }
const DETECTION_BOX = { x1: 11, y1: 13, x2: 69, y2: 80 }

function matchingObservation(confidence = 0.8) {
    return { detection_box: DETECTION_BOX, confidence, iou_with_ground_truth: 0.9 }
}

function transitionFixture() {
    return {
        dataset_id: 'dataset-1',
        results: [
            { image_id: 'B', filename: 'other.jpg', adversarial_image_url: '/other.png' },
            { image_id: 'A', filename: 'selected.jpg', adversarial_image_url: '/selected.png' },
        ],
        supervised: { object_analysis: {
            objects: [
                {
                    object_id: 'different-image', image_id: 'B',
                    category: { category_name: 'Otro' }, ground_truth_box: BOX,
                    original_observation: matchingObservation(), adversarial_observation: null,
                },
                ...[
                    ['preserved', matchingObservation(), matchingObservation(0.6)],
                    ['suppressed', matchingObservation(), null],
                    ['recovered', null, matchingObservation(0.6)],
                    ['missed_both', null, null],
                ].map(([outcome, original, adversarial]) => ({
                    object_id: outcome, image_id: 'A', outcome,
                    category: { category_name: 'Coche' }, ground_truth_box: BOX,
                    original_observation: original, adversarial_observation: adversarial,
                })),
            ],
            by_image: [{ image_id: 'A', total_objects: 4 }],
        } },
    }
}

test('el modo objeto aísla únicamente el objeto seleccionado en la imagen correcta', () => {
    const evaluation = transitionFixture()
    const preview = getDetectionObjectImagePreview(evaluation, 'A', 'suppressed')

    assert.equal(preview.filename, 'selected.jpg')
    assert.equal(preview.selectedObject.object_id, 'suppressed')
    assert.equal(preview.objects.length, 4)
    assert.equal(preview.originalDetections.length, 2)
    assert.equal(preview.adversarialDetections.length, 1)
    assert.equal(preview.originalDetections[0].tone, 'reference')
    assert.equal(preview.adversarialDetections[0].tone, 'reference')
    assert.equal(preview.originalDetections[1].tone, 'original')
    assert.equal(preview.adversarialImagePath, '/selected.png')
})

test('preservado, suprimido, recuperado y no detectado muestran solo los matches existentes', () => {
    const evaluation = transitionFixture()
    const expected = [
        ['preserved', ['reference', 'original'], ['reference', 'adversarial']],
        ['suppressed', ['reference', 'original'], ['reference']],
        ['recovered', ['reference'], ['reference', 'adversarial']],
        ['missed_both', ['reference'], ['reference']],
    ]

    for (const [objectId, originalTones, adversarialTones] of expected) {
        const preview = getDetectionObjectImagePreview(evaluation, 'A', objectId)
        assert.deepEqual(preview.originalDetections.map((box) => box.tone), originalTones, objectId)
        assert.deepEqual(preview.adversarialDetections.map((box) => box.tone), adversarialTones, objectId)
    }
})

test('la referencia COCO muestra solo la clase con su estilo de referencia', () => {
    const preview = getDetectionObjectImagePreview(transitionFixture(), 'A')
    assert.equal(preview.selectedObject, null)
    assert.equal(preview.originalDetections.length, 6) // 4 referencias + 2 matches
    assert.equal(preview.adversarialDetections.length, 6)
    assert.equal(preview.originalDetections.filter((item) => item.tone === 'reference').length, 4)
    assert.ok(preview.originalDetections.filter((item) => item.tone === 'reference')
        .every((item) => item.label === 'Coche' && item.showLabel !== false && item.confidence == null))
    assert.ok(preview.adversarialDetections.filter((item) => item.tone === 'reference')
        .every((item) => item.label === 'Coche' && item.showLabel !== false && item.confidence == null))
    assert.ok(preview.originalDetections.filter((item) => item.tone !== 'reference')
        .every((item) => item.showLabel !== false))
})

test('un ID de objeto de otra imagen nunca selecciona un objeto ajeno', () => {
    const preview = getDetectionObjectImagePreview(transitionFixture(), 'A', 'different-image')
    assert.equal(preview.selectedObject, null)
    assert.deepEqual(preview.originalDetections, [])
    assert.deepEqual(preview.adversarialDetections, [])
})

test('los resultados históricos sin imagen adversaria aún permiten aislar la referencia', () => {
    const evaluation = transitionFixture()
    evaluation.results = []
    const preview = getDetectionObjectImagePreview(evaluation, 'A', 'missed_both')
    assert.equal(preview.adversarialImagePath, null)
    assert.equal(preview.selectedObject.outcome, 'missed_both')
    assert.equal(preview.originalDetections.length, 1)
    assert.equal(preview.adversarialDetections.length, 1)
})
