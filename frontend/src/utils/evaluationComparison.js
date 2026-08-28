function normalizedValue(value) {
    if (Array.isArray(value)) return value.map(normalizedValue)
    if (value && typeof value === 'object') {
        return Object.fromEntries(
            Object.entries(value)
                .filter(([, entry]) => entry !== undefined)
                .sort(([left], [right]) => left.localeCompare(right))
                .map(([key, entry]) => [key, normalizedValue(entry)]),
        )
    }
    return value
}

export function stableComparisonKey(value) {
    return JSON.stringify(normalizedValue(value ?? null))
}

function sameValue(left, right) {
    return stableComparisonKey(left) === stableComparisonKey(right)
}

function invalid(reason, code) {
    return {
        compatible: false,
        kind: null,
        code,
        reason,
    }
}

export function classifyEvaluationComparison(left, right) {
    if (!left || !right) {
        return invalid('Selecciona dos evaluaciones para compararlas.', 'missing')
    }
    if (left.result_id === right.result_id) {
        return invalid('Selecciona dos evaluaciones diferentes.', 'same_result')
    }
    if (left.status !== 'completed' || right.status !== 'completed') {
        return invalid('Solo se pueden comparar evaluaciones completadas.', 'not_completed')
    }
    if (left.task_type !== right.task_type) {
        return invalid(
            'Las evaluaciones deben pertenecer a la misma tarea.',
            'different_task',
        )
    }
    if (left.dataset_id !== right.dataset_id) {
        return invalid(
            'Las evaluaciones deben utilizar el mismo conjunto de datos.',
            'different_dataset',
        )
    }
    if (left.dataset_revision !== right.dataset_revision) {
        return invalid(
            'El conjunto de datos cambió entre ambas evaluaciones. Compara resultados de la misma revisión.',
            'different_dataset_revision',
        )
    }
    if (!sameValue(left.supervised_configuration, right.supervised_configuration)) {
        return invalid(
            'El protocolo supervisado de evaluación debe ser idéntico.',
            'different_supervised_protocol',
        )
    }

    const sameModel = left.model_id === right.model_id
    const sameAttack = left.attack_type === right.attack_type
    const sameParameters = sameValue(left.parameters, right.parameters)

    if (sameModel && !sameAttack) {
        return {
            compatible: true,
            kind: 'attack',
            code: 'attack',
            label: 'Ataques',
            description: 'Mismo modelo y conjunto de datos; cambia el ataque aplicado.',
        }
    }

    if (!sameModel && sameAttack && sameParameters) {
        return {
            compatible: true,
            kind: 'model',
            code: 'model',
            label: 'Modelos',
            description: 'Mismo ataque, parámetros y conjunto de datos; cambia el modelo.',
        }
    }

    if (sameModel && sameAttack && !sameParameters) {
        return {
            compatible: true,
            kind: 'attack_configuration',
            code: 'attack_configuration',
            label: 'Configuración del ataque',
            description: 'Mismo modelo y ataque; cambia únicamente su configuración.',
        }
    }

    if (sameModel && sameAttack && sameParameters) {
        return invalid(
            'Las dos evaluaciones tienen la misma configuración experimental.',
            'identical_configuration',
        )
    }

    return invalid(
        'Cambian varias variables a la vez. Selecciona una comparación controlada de modelo, ataque o configuración.',
        'multiple_variables',
    )
}

export function getCompatibleEvaluationCandidates(base, items) {
    return (items ?? [])
        .map((item) => ({
            item,
            comparison: classifyEvaluationComparison(base, item),
        }))
        .filter(({ comparison }) => comparison.compatible)
}

function finiteMetric(value) {
    const candidate = value?.available === true ? value.value : value
    return typeof candidate === 'number' && Number.isFinite(candidate)
        ? candidate
        : null
}

function metric(key, label, value, options = {}) {
    return {
        key,
        label,
        value: finiteMetric(value),
        format: options.format ?? 'ratio',
        higherIsBetter: options.higherIsBetter ?? true,
    }
}

export function getClassificationComparisonMetrics(evaluation) {
    const robustness = evaluation?.advanced_metrics?.robustness
    const aggregate = evaluation?.advanced_metrics?.supervised?.aggregate
    const summary = evaluation?.summary

    return [
        metric(
            'clean_accuracy',
            'Exactitud original',
            robustness?.clean_accuracy ?? summary?.original_accuracy,
        ),
        metric(
            'robust_accuracy',
            'Exactitud robusta',
            robustness?.robust_accuracy ?? summary?.adversarial_accuracy,
        ),
        metric(
            'accuracy_drop',
            'Caída de exactitud',
            robustness?.accuracy_drop ?? summary?.accuracy_drop,
            { format: 'points', higherIsBetter: false },
        ),
        metric(
            'conditional_asr',
            'ASR condicional',
            robustness?.conditional_attack_success_rate,
            { higherIsBetter: false },
        ),
        metric(
            'robust_f1_macro',
            'F1 macro robusto',
            aggregate?.adversarial?.f1_macro,
        ),
    ]
}

export function getDetectionComparisonMetrics(evaluation) {
    const supervised = evaluation?.supervised
    const originalAggregate = supervised?.original?.aggregate
    const aggregate = supervised?.adversarial?.aggregate
    const deltas = supervised?.deltas?.aggregate
    const operatingPoint = supervised?.adversarial?.operating_point
    const objectSummary = supervised?.object_analysis?.summary

    return [
        metric('original_map', 'mAP original', originalAggregate?.map_50_95),
        metric('robust_map', 'mAP robusto', aggregate?.map_50_95),
        metric('map_drop', 'Degradación de mAP', deltas?.map_50_95?.absolute, {
            format: 'points',
            higherIsBetter: false,
        }),
        metric('robust_ap50', 'AP@0.50 robusto', aggregate?.ap_50),
        metric('robust_ar100', 'AR@100 robusto', aggregate?.ar_100),
        metric('robust_f1', 'F1 robusto', operatingPoint?.f1),
        metric('suppression_rate', 'Supresión de objetos', objectSummary?.suppression_rate, {
            higherIsBetter: false,
        }),
    ]
}

export function getEvaluationComparisonMetrics(taskType, evaluation) {
    return taskType === 'detection'
        ? getDetectionComparisonMetrics(evaluation)
        : getClassificationComparisonMetrics(evaluation)
}

export function pairComparisonMetrics(taskType, leftEvaluation, rightEvaluation) {
    const leftMetrics = getEvaluationComparisonMetrics(taskType, leftEvaluation)
    const rightByKey = new Map(
        getEvaluationComparisonMetrics(taskType, rightEvaluation).map((entry) => [
            entry.key,
            entry,
        ]),
    )

    return leftMetrics.map((left) => {
        const right = rightByKey.get(left.key) ?? { ...left, value: null }
        const delta = left.value != null && right.value != null
            ? right.value - left.value
            : null
        const advantage = delta == null || Math.abs(delta) < 1e-12
            ? 'equal'
            : (delta > 0) === left.higherIsBetter
                ? 'right'
                : 'left'

        return {
            ...left,
            left: left.value,
            right: right.value,
            delta,
            advantage,
        }
    })
}

export function baselineConsistency(taskType, leftEvaluation, rightEvaluation) {
    const leftMetric = taskType === 'detection'
        ? leftEvaluation?.supervised?.original?.aggregate?.map_50_95
        : leftEvaluation?.advanced_metrics?.robustness?.clean_accuracy
            ?? leftEvaluation?.summary?.original_accuracy
    const rightMetric = taskType === 'detection'
        ? rightEvaluation?.supervised?.original?.aggregate?.map_50_95
        : rightEvaluation?.advanced_metrics?.robustness?.clean_accuracy
            ?? rightEvaluation?.summary?.original_accuracy
    const left = finiteMetric(leftMetric)
    const right = finiteMetric(rightMetric)

    if (left == null || right == null) {
        return { available: false, consistent: null, difference: null }
    }

    const difference = Math.abs(left - right)
    return {
        available: true,
        consistent: difference <= 1e-9,
        difference,
    }
}

function comparisonMetricValue(metricValue) {
    return finiteMetric(metricValue)
}

export function getClassificationClassComparison(leftEvaluation, rightEvaluation, metricId = 'f1') {
    const leftRows = leftEvaluation?.advanced_metrics?.supervised?.per_class ?? []
    const rightRows = rightEvaluation?.advanced_metrics?.supervised?.per_class ?? []
    const rightByLabel = new Map(
        rightRows.map((row) => [String(row.class_reference?.label ?? '').toLocaleLowerCase('es'), row]),
    )

    return leftRows.map((left) => {
        const label = left.class_reference?.label ?? 'Clase sin nombre'
        const right = rightByLabel.get(String(label).toLocaleLowerCase('es'))
        const leftRobust = comparisonMetricValue(left.adversarial?.[metricId])
        const rightRobust = comparisonMetricValue(right?.adversarial?.[metricId])
        return {
            key: `${left.class_reference?.index ?? label}`,
            label,
            support: Number(left.support ?? 0),
            leftOriginal: comparisonMetricValue(left.original?.[metricId]),
            leftRobust,
            rightOriginal: comparisonMetricValue(right?.original?.[metricId]),
            rightRobust,
            delta: leftRobust != null && rightRobust != null ? rightRobust - leftRobust : null,
        }
    })
}

function detectionClassKey(row) {
    return String(
        row?.reference?.category_internal_id
        ?? row?.reference?.category_name
        ?? row?.reference?.model_class_name
        ?? '',
    )
}

export function getDetectionClassComparison(leftEvaluation, rightEvaluation) {
    const leftSupervised = leftEvaluation?.supervised
    const rightSupervised = rightEvaluation?.supervised
    const rightOriginal = new Map(
        (rightSupervised?.original?.per_class ?? []).map((row) => [detectionClassKey(row), row]),
    )
    const rightAdversarial = new Map(
        (rightSupervised?.adversarial?.per_class ?? []).map((row) => [detectionClassKey(row), row]),
    )
    const leftAdversarial = new Map(
        (leftSupervised?.adversarial?.per_class ?? []).map((row) => [detectionClassKey(row), row]),
    )
    const leftObjects = new Map(
        (leftSupervised?.object_analysis?.by_class ?? []).map((row) => [
            String(row.category?.category_internal_id ?? row.category?.category_name ?? ''),
            row,
        ]),
    )
    const rightObjects = new Map(
        (rightSupervised?.object_analysis?.by_class ?? []).map((row) => [
            String(row.category?.category_internal_id ?? row.category?.category_name ?? ''),
            row,
        ]),
    )

    return (leftSupervised?.original?.per_class ?? []).map((leftOriginalRow) => {
        const key = detectionClassKey(leftOriginalRow)
        const rightOriginalRow = rightOriginal.get(key)
        const leftAdversarialRow = leftAdversarial.get(key)
        const rightAdversarialRow = rightAdversarial.get(key)
        const leftAp = comparisonMetricValue(leftAdversarialRow?.average_precision)
        const rightAp = comparisonMetricValue(rightAdversarialRow?.average_precision)
        return {
            key,
            label: leftOriginalRow.reference?.category_name ?? leftOriginalRow.reference?.model_class_name ?? 'Clase',
            support: Number(leftOriginalRow.support ?? rightOriginalRow?.support ?? 0),
            leftAp,
            rightAp,
            leftAr: comparisonMetricValue(leftAdversarialRow?.average_recall),
            rightAr: comparisonMetricValue(rightAdversarialRow?.average_recall),
            leftSuppressed: Number(leftObjects.get(key)?.suppressed_count ?? 0),
            rightSuppressed: Number(rightObjects.get(key)?.suppressed_count ?? 0),
            leftPreservationRate: comparisonMetricValue(leftObjects.get(key)?.preservation_rate),
            rightPreservationRate: comparisonMetricValue(rightObjects.get(key)?.preservation_rate),
            leftSuppressionRate: comparisonMetricValue(leftObjects.get(key)?.suppression_rate),
            rightSuppressionRate: comparisonMetricValue(rightObjects.get(key)?.suppression_rate),
            delta: leftAp != null && rightAp != null ? rightAp - leftAp : null,
        }
    })
}

export function getDetectionCocoComparisonRows(leftEvaluation, rightEvaluation) {
    const fields = [
        ['map_50_95', 'mAP@[0.50:0.95]'],
        ['ap_50', 'AP@0.50'],
        ['ap_75', 'AP@0.75'],
        ['ar_1', 'AR@1'],
        ['ar_10', 'AR@10'],
        ['ar_100', 'AR@100'],
        ['ap_small', 'AP pequeño'],
        ['ap_medium', 'AP mediano'],
        ['ap_large', 'AP grande'],
        ['ar_small', 'AR pequeño'],
        ['ar_medium', 'AR mediano'],
        ['ar_large', 'AR grande'],
    ]
    const left = leftEvaluation?.supervised?.adversarial?.aggregate ?? {}
    const right = rightEvaluation?.supervised?.adversarial?.aggregate ?? {}

    return fields.map(([key, label]) => {
        const leftValue = comparisonMetricValue(left[key])
        const rightValue = comparisonMetricValue(right[key])
        return {
            key,
            label,
            left: leftValue,
            right: rightValue,
            delta: leftValue != null && rightValue != null ? rightValue - leftValue : null,
        }
    })
}

export function getDetectionTideComparisonRows(leftEvaluation, rightEvaluation) {
    const left = leftEvaluation?.supervised?.tide_analysis
    const right = rightEvaluation?.supervised?.tide_analysis
    if (!left || !right) return []

    const rightMain = new Map((right.adversarial?.main_errors ?? []).map((row) => [row.error_type, row]))
    return (left.adversarial?.main_errors ?? []).map((leftRow) => {
        const rightRow = rightMain.get(leftRow.error_type)
        return {
            key: leftRow.error_type,
            leftImpact: Number(leftRow.delta_ap ?? 0),
            rightImpact: rightRow ? Number(rightRow.delta_ap ?? 0) : null,
            leftCount: Number(leftRow.count ?? 0),
            rightCount: rightRow ? Number(rightRow.count ?? 0) : null,
        }
    })
}


export function getComparisonRetention(taskType, evaluation) {
    const original = taskType === 'detection'
        ? finiteMetric(evaluation?.supervised?.original?.aggregate?.map_50_95)
        : finiteMetric(evaluation?.advanced_metrics?.robustness?.clean_accuracy ?? evaluation?.summary?.original_accuracy)
    const robust = taskType === 'detection'
        ? finiteMetric(evaluation?.supervised?.adversarial?.aggregate?.map_50_95)
        : finiteMetric(evaluation?.advanced_metrics?.robustness?.robust_accuracy ?? evaluation?.summary?.adversarial_accuracy)

    return {
        original,
        robust,
        retention: original != null && original > 0 && robust != null ? robust / original : null,
    }
}

function classificationCorrect(result) {
    if (typeof result?.adversarial_correct === 'boolean') return result.adversarial_correct
    if (result?.ground_truth_label == null || result?.adversarial_label == null) return null
    return String(result.ground_truth_label) === String(result.adversarial_label)
}

function matchesDetectionImageFilter(left, right, filter) {
    const leftCount = Number(left.adversarial_detection_count ?? 0)
    const rightCount = Number(right.adversarial_detection_count ?? 0)
    if (filter === 'left_more') return leftCount > rightCount
    if (filter === 'right_more') return rightCount > leftCount
    if (filter === 'different') return leftCount !== rightCount
    if (filter === 'equal') return leftCount === rightCount
    return true
}

function matchesClassificationImageFilter(left, right, filter) {
    const leftCorrect = classificationCorrect(left)
    const rightCorrect = classificationCorrect(right)
    if (leftCorrect == null || rightCorrect == null) return false
    if (filter === 'left_better') return leftCorrect && !rightCorrect
    if (filter === 'right_better') return rightCorrect && !leftCorrect
    if (filter === 'both_fail') return !leftCorrect && !rightCorrect
    if (filter === 'both_correct') return leftCorrect && rightCorrect
    return true
}

export function matchesComparisonImageFilter(taskType, pair, filter = 'all') {
    if (filter === 'all') return true
    const left = pair?.left
    const right = pair?.right
    if (!left || !right) return false

    return taskType === 'detection'
        ? matchesDetectionImageFilter(left, right, filter)
        : matchesClassificationImageFilter(left, right, filter)
}

export function getComparisonImagePairs(leftEvaluation, rightEvaluation) {
    const leftResults = leftEvaluation?.results ?? []
    const rightById = new Map((rightEvaluation?.results ?? []).map((result) => [String(result.image_id), result]))
    return leftResults
        .map((left) => ({ left, right: rightById.get(String(left.image_id)) }))
        .filter((pair) => pair.right)
}

export function getDetectionObjectComparisonPairs(leftEvaluation, rightEvaluation) {
    const left = leftEvaluation?.supervised?.object_analysis?.objects ?? []
    const rightById = new Map(
        (rightEvaluation?.supervised?.object_analysis?.objects ?? []).map((object) => [String(object.object_id), object]),
    )
    return left
        .map((leftObject) => ({ left: leftObject, right: rightById.get(String(leftObject.object_id)) }))
        .filter((pair) => pair.right)
}

function robustComparisonInsight(taskType, metricValue) {
    if (metricValue?.delta == null || Math.abs(metricValue.delta) < 1e-12) return null
    const favored = metricValue.delta > 0 ? 'B' : 'A'
    const metricLabel = taskType === 'detection' ? 'mAP' : 'exactitud'
    return {
        key: 'robust',
        side: favored.toLowerCase(),
        value: Math.abs(metricValue.delta),
        text: `${favored} conserva ${Math.abs(metricValue.delta * 100).toFixed(1)} pp más de ${metricLabel} tras el ataque.`,
    }
}

function dropComparisonInsight(metricValue) {
    if (metricValue?.delta == null || Math.abs(metricValue.delta) < 1e-12) return null
    const lowerSide = metricValue.delta > 0 ? 'A' : 'B'
    return {
        key: 'drop',
        side: lowerSide.toLowerCase(),
        value: Math.abs(metricValue.delta),
        text: `${lowerSide} sufre ${Math.abs(metricValue.delta * 100).toFixed(1)} pp menos de degradación respecto a su referencia original.`,
    }
}

function strongestClassComparisonInsight(classRows) {
    const strongestClass = [...(classRows ?? [])]
        .filter((row) => Number.isFinite(row.delta))
        .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))[0]
    if (!strongestClass || Math.abs(strongestClass.delta) < 1e-12) return null
    const favored = strongestClass.delta > 0 ? 'B' : 'A'
    return {
        key: 'class',
        side: favored.toLowerCase(),
        value: Math.abs(strongestClass.delta),
        text: `La mayor separación por clase aparece en «${strongestClass.label}»: ${favored} queda ${Math.abs(strongestClass.delta * 100).toFixed(1)} pp por encima.`,
    }
}


export function buildClassificationCurveSeries(left, right, sameModel) {
    const series = []
    if (sameModel && left?.original?.available) {
        series.push({
            key: 'reference',
            label: 'Referencia original común',
            color: 'var(--comparison-reference)',
            reference: true,
            points: left.original.points,
            score: left.original.score,
        })
    }
    if (!sameModel && left?.original?.available) {
        series.push({
            key: 'a-original',
            label: 'A · original',
            color: 'var(--comparison-a)',
            reference: true,
            points: left.original.points,
            score: left.original.score,
        })
    }
    if (!sameModel && right?.original?.available) {
        series.push({
            key: 'b-original',
            label: 'B · original',
            color: 'var(--comparison-b)',
            reference: true,
            points: right.original.points,
            score: right.original.score,
        })
    }
    if (left?.adversarial?.available) {
        series.push({
            key: 'a',
            label: 'A · robusta',
            color: 'var(--comparison-a)',
            points: left.adversarial.points,
            score: left.adversarial.score,
        })
    }
    if (right?.adversarial?.available) {
        series.push({
            key: 'b',
            label: 'B · robusta',
            color: 'var(--comparison-b)',
            points: right.adversarial.points,
            score: right.adversarial.score,
        })
    }
    return series
}

export function getComparisonInsights(taskType, metrics, classRows = []) {
    const byKey = new Map((metrics ?? []).map((entry) => [entry.key, entry]))
    const robustKey = taskType === 'detection' ? 'robust_map' : 'robust_accuracy'
    const dropKey = taskType === 'detection' ? 'map_drop' : 'accuracy_drop'

    return [
        robustComparisonInsight(taskType, byKey.get(robustKey)),
        dropComparisonInsight(byKey.get(dropKey)),
        strongestClassComparisonInsight(classRows),
    ].filter(Boolean).slice(0, 3)
}

function exactTwoSidedBinomialP(leftCount, rightCount) {
    const total = leftCount + rightCount
    if (total === 0) return 1

    const tail = Math.min(leftCount, rightCount)
    let probability = 2 ** (-total)
    let cumulative = probability
    for (let successes = 0; successes < tail; successes += 1) {
        probability *= (total - successes) / (successes + 1)
        cumulative += probability
    }
    return Math.min(1, cumulative * 2)
}

export function getClassificationComparisonCaseSummary(leftEvaluation, rightEvaluation) {
    const pairs = getComparisonImagePairs(leftEvaluation, rightEvaluation)
    const leftBetter = pairs.filter((pair) => matchesComparisonImageFilter('classification', pair, 'left_better')).length
    const rightBetter = pairs.filter((pair) => matchesComparisonImageFilter('classification', pair, 'right_better')).length
    const discordant = leftBetter + rightBetter
    return {
        total: pairs.length,
        left_better: leftBetter,
        right_better: rightBetter,
        both_fail: pairs.filter((pair) => matchesComparisonImageFilter('classification', pair, 'both_fail')).length,
        both_correct: pairs.filter((pair) => matchesComparisonImageFilter('classification', pair, 'both_correct')).length,
        discordant,
        disagreement_rate: pairs.length ? discordant / pairs.length : 0,
        mcnemar_exact_p: exactTwoSidedBinomialP(leftBetter, rightBetter),
    }
}

export function getDetectionImageCaseSummary(leftEvaluation, rightEvaluation) {
    const pairs = getComparisonImagePairs(leftEvaluation, rightEvaluation)
    return {
        total: pairs.length,
        left_more: pairs.filter((pair) => matchesComparisonImageFilter('detection', pair, 'left_more')).length,
        right_more: pairs.filter((pair) => matchesComparisonImageFilter('detection', pair, 'right_more')).length,
        different: pairs.filter((pair) => matchesComparisonImageFilter('detection', pair, 'different')).length,
        equal: pairs.filter((pair) => matchesComparisonImageFilter('detection', pair, 'equal')).length,
    }
}

export function matchesDetectionObjectCaseFilter(pair, filter = 'all') {
    if (filter === 'all') return true
    const leftOutcome = pair?.left?.outcome
    const rightOutcome = pair?.right?.outcome
    if (!leftOutcome || !rightOutcome) return false
    if (filter === 'only_a') return leftOutcome === 'preserved' && rightOutcome !== 'preserved'
    if (filter === 'only_b') return rightOutcome === 'preserved' && leftOutcome !== 'preserved'
    if (filter === 'both_preserved') return leftOutcome === 'preserved' && rightOutcome === 'preserved'
    if (filter === 'both_suppressed') return leftOutcome === 'suppressed' && rightOutcome === 'suppressed'
    if (filter === 'different') return leftOutcome !== rightOutcome
    return true
}

export function getDetectionObjectCaseSummary(leftEvaluation, rightEvaluation) {
    const pairs = getDetectionObjectComparisonPairs(leftEvaluation, rightEvaluation)
    const onlyA = pairs.filter((pair) => matchesDetectionObjectCaseFilter(pair, 'only_a')).length
    const onlyB = pairs.filter((pair) => matchesDetectionObjectCaseFilter(pair, 'only_b')).length
    const preservationDisagreements = onlyA + onlyB
    return {
        total: pairs.length,
        only_a: onlyA,
        only_b: onlyB,
        both_preserved: pairs.filter((pair) => matchesDetectionObjectCaseFilter(pair, 'both_preserved')).length,
        both_suppressed: pairs.filter((pair) => matchesDetectionObjectCaseFilter(pair, 'both_suppressed')).length,
        different: pairs.filter((pair) => matchesDetectionObjectCaseFilter(pair, 'different')).length,
        preservation_disagreements: preservationDisagreements,
        preservation_disagreement_rate: pairs.length ? preservationDisagreements / pairs.length : 0,
        preservation_balance: onlyB - onlyA,
    }
}
