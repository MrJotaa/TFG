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
