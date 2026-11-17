export const DETECTION_OBJECT_PAGE_SIZES = [25, 50, 100]

function normalize(value) {
    return String(value ?? '').trim().toLocaleLowerCase('es')
}

export function getDetectionObjectCategoryKey(object) {
    return String(
        object?.category?.category_internal_id
            ?? object?.category?.model_class_id
            ?? object?.category?.category_name
            ?? '',
    )
}

export function getDetectionObjectFilterOptions(objects = []) {
    const images = new Set()
    const categories = new Map()

    objects.forEach((object) => {
        images.add(String(object.image_id))
        const key = getDetectionObjectCategoryKey(object)
        if (!categories.has(key)) {
            categories.set(key, object.category?.category_name ?? key)
        }
    })

    return {
        images: [...images].sort((left, right) => left.localeCompare(right, 'es')),
        categories: [...categories]
            .map(([value, label]) => ({ value, label }))
            .sort((left, right) => left.label.localeCompare(right.label, 'es')),
    }
}

function compareNullableNumbers(left, right, direction) {
    const leftNumber = Number(left)
    const rightNumber = Number(right)
    const leftAvailable = left !== null && left !== undefined && Number.isFinite(leftNumber)
    const rightAvailable = right !== null && right !== undefined && Number.isFinite(rightNumber)

    if (!leftAvailable && !rightAvailable) {
        return 0
    }
    if (!leftAvailable) {
        return 1
    }
    if (!rightAvailable) {
        return -1
    }

    return direction === 'asc' ? leftNumber - rightNumber : rightNumber - leftNumber
}

function compareObjects(left, right, sortBy) {
    const numericSorts = {
        confidence_drop: [left.confidence_drop, right.confidence_drop, 'desc'],
        iou_drop: [left.iou_drop, right.iou_drop, 'desc'],
        adversarial_confidence: [
            left.adversarial_observation?.confidence,
            right.adversarial_observation?.confidence,
            'asc',
        ],
        adversarial_iou: [
            left.adversarial_observation?.iou_with_ground_truth,
            right.adversarial_observation?.iou_with_ground_truth,
            'asc',
        ],
    }

    if (numericSorts[sortBy]) {
        const comparison = compareNullableNumbers(...numericSorts[sortBy])
        if (comparison !== 0) {
            return comparison
        }
    }

    if (sortBy === 'class') {
        const comparison = String(left.category?.category_name ?? '').localeCompare(
            String(right.category?.category_name ?? ''),
            'es',
        )
        if (comparison !== 0) {
            return comparison
        }
    }

    return `${left.image_id}:${left.object_id}`.localeCompare(
        `${right.image_id}:${right.object_id}`,
        'es',
    )
}

export function filterAndSortDetectionObjects(objects = [], filters = {}) {
    const query = normalize(filters.query)

    return objects
        .filter((object) => {
            if (filters.outcome && filters.outcome !== 'all' && object.outcome !== filters.outcome) {
                return false
            }
            if (filters.imageId && filters.imageId !== 'all' && String(object.image_id) !== filters.imageId) {
                return false
            }
            if (filters.categoryKey && filters.categoryKey !== 'all' && getDetectionObjectCategoryKey(object) !== filters.categoryKey) {
                return false
            }
            if (!query) {
                return true
            }

            return [
                object.object_id,
                object.image_id,
                object.category?.category_name,
                object.category?.model_class_name,
            ].some((value) => normalize(value).includes(query))
        })
        .sort((left, right) => compareObjects(left, right, filters.sortBy ?? 'canonical'))
}

export function paginateDetectionObjects(objects = [], page = 1, pageSize = 25) {
    const normalizedSize = DETECTION_OBJECT_PAGE_SIZES.includes(Number(pageSize))
        ? Number(pageSize)
        : DETECTION_OBJECT_PAGE_SIZES[0]
    const totalPages = Math.max(1, Math.ceil(objects.length / normalizedSize))
    const safePage = Math.min(Math.max(1, Number(page) || 1), totalPages)
    const start = (safePage - 1) * normalizedSize
    const end = Math.min(start + normalizedSize, objects.length)

    return {
        items: objects.slice(start, end),
        page: safePage,
        pageSize: normalizedSize,
        totalPages,
        start,
        end,
        totalItems: objects.length,
    }
}

export const DETECTION_IMAGE_AGGREGATE_BATCH_SIZE = 10

export function getDetectionImageAggregateView(
    rows = [],
    visibleCount = DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
) {
    const safeVisibleCount = Math.max(
        DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
        Number(visibleCount) || DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
    )
    const end = Math.min(safeVisibleCount, rows.length)

    return {
        items: rows.slice(0, end),
        visibleCount: end,
        totalCount: rows.length,
        hasMore: end < rows.length,
        nextVisibleCount: Math.min(
            rows.length,
            end + DETECTION_IMAGE_AGGREGATE_BATCH_SIZE,
        ),
    }
}

// The object-level view and the image result list use the same evaluation
// snapshot. Match by ID, not by list index: order can differ after filtering.
function isRenderableBox(box) {
    if (!box) return false
    const { x1, y1, x2, y2 } = box
    return [x1, y1, x2, y2].every((value) => typeof value === 'number' && Number.isFinite(value))
        && x2 > x1 && y2 > y1
}

export function getDetectionObjectImagePreview(evaluation, imageId, objectId = null) {
    const key = String(imageId)
    const result = (evaluation?.results ?? []).find((item) => String(item.image_id) === key)
    const objects = (evaluation?.supervised?.object_analysis?.objects ?? [])
        .filter((item) => String(item.image_id) === key)
    const aggregate = (evaluation?.supervised?.object_analysis?.by_image ?? [])
        .find((item) => String(item.image_id) === key)
    // A selected object belongs to this image: never look it up globally by ID
    // or silently fall back to an unrelated object when a record is missing.
    const selectedObject = objectId == null
        ? null
        : objects.find((item) => String(item.object_id) === String(objectId)) ?? null

    function buildOverlay(phase) {
        const detections = []
        for (const object of selectedObject ? [selectedObject] : objectId == null ? objects : []) {
            const category = object.category?.category_name ?? 'Objeto'
            if (isRenderableBox(object.ground_truth_box)) {
                detections.push({
                    bbox: object.ground_truth_box,
                    label: category,
                    // Show only the class name: reference styling and the legend
                    // distinguish this annotation from model detections.
                    tone: 'reference',
                })
            }
            const observation = phase === 'original'
                ? object.original_observation
                : object.adversarial_observation
            if (isRenderableBox(observation?.detection_box)) {
                detections.push({
                    bbox: observation.detection_box,
                    label: category,
                    confidence: observation.confidence,
                    tone: phase,
                })
            }
        }
        return detections
    }

    return {
        imageId: key,
        filename: result?.filename ?? null,
        originalImagePath: evaluation?.dataset_id != null
            ? `/datasets/${encodeURIComponent(String(evaluation.dataset_id))}/images/${encodeURIComponent(key)}`
            : null,
        adversarialImagePath: result?.adversarial_image_url || null,
        aggregate: aggregate ?? null,
        objects,
        selectedObject,
        originalDetections: buildOverlay('original'),
        adversarialDetections: buildOverlay('adversarial'),
    }
}
