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
