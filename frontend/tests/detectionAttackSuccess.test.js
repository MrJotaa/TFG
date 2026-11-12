import assert from 'node:assert/strict'
import test from 'node:test'

import {
    detectionSuccessLabel,
    detectionOutcomeExplanation,
    getDetectionSuccessPresentation,
    squareObjectivePresentation,
} from '../src/utils/detectionAttackSuccess.js'
import { getDetectionMetricGuide } from '../src/utils/evaluationMetricGuides.js'
import { historySummary } from '../src/utils/evaluationHistory.js'
import { individualAnalysisOutcome, classifyIndividualAnalysisComparison } from '../src/utils/individualAnalysisHistory.js'
import { individualAnalysisComparisonSummary } from '../src/utils/individualAnalysisComparison.js'

const NEW_CRITERION = {
    metrics_schema_version: '1.1',
    attack_success_criterion: 'matched_detections_iou50_v1',
}
const OLD_CRITERION = {
    metrics_schema_version: '1.0',
    attack_success_criterion: 'legacy_count_or_confidence_v1',
}

test('la ausencia de metadatos históricos NUNCA implica reinterpretar attack_success con v1.1', () => {
    assert.equal(getDetectionSuccessPresentation({}).version, '1.0')
    assert.equal(getDetectionSuccessPresentation(OLD_CRITERION).version, '1.0')
    assert.equal(detectionSuccessLabel({}, true), 'Cambio según criterio histórico')
    assert.equal(detectionSuccessLabel({}, false), 'Criterio histórico no cumplido')
    assert.match(getDetectionSuccessPresentation({}).description, /confianza media/)
})

test('v1.1 declara correspondencia por clase e IoU sin confundirla con COCO', () => {
    const copy = getDetectionSuccessPresentation(NEW_CRITERION)
    assert.equal(copy.version, '1.1')
    assert.match(copy.description, /IoU ≥ 0,50/)
    assert.match(copy.description, /no con las anotaciones reales/)
    assert.match(copy.description, /bajada aislada de confianza no cuenta/)
    assert.equal(detectionSuccessLabel(NEW_CRITERION, true), 'Alteración de detecciones')
    assert.equal(detectionSuccessLabel(NEW_CRITERION, false), 'Correspondencias conservadas')
    assert.equal(detectionSuccessLabel(NEW_CRITERION, null), 'Resultado sin indicador de éxito')
})

test('la versión de una entrada guardada se obtiene de su resumen sin nuevas peticiones', () => {
    const entry = { task_type: 'detection', summary: { ...NEW_CRITERION, attack_success: true } }
    assert.equal(individualAnalysisOutcome(entry), 'Alteración de detecciones')
    assert.equal(individualAnalysisOutcome({ ...entry, summary: { ...OLD_CRITERION, attack_success: true } }), 'Cambio según criterio histórico')
    assert.equal(individualAnalysisOutcome({ task_type: 'classification', summary: { attack_success: true } }), 'Ataque exitoso')
})

test('resumen de evaluación histórica identifica criterio cuando existe y mantiene formato antiguo sin metadatos', () => {
    const data = { task_type: 'detection', summary: { total_original_detections: 12, total_adversarial_detections: 8 } }
    assert.equal(historySummary(data), '12 → 8 detecciones')
    assert.match(historySummary({ ...data, ...NEW_CRITERION }), /v1\.1/)
    assert.match(historySummary({ ...data, ...OLD_CRITERION }), /v1\.0/)
})

test('la guía de pestañas se adapta a la definición almacenada sin afectar a clasificación', () => {
    const modern = getDetectionMetricGuide('images', NEW_CRITERION)
    const legacy = getDetectionMetricGuide('images', OLD_CRITERION)
    assert.match(modern.items.map(({ definition }) => definition).join(' '), /emparejamiento|correspondencia/)
    assert.match(legacy.items.map(({ definition }) => definition).join(' '), /confianza media/)
    assert.notEqual(modern.items[2].term, legacy.items[2].term)
    assert.match(getDetectionMetricGuide('overview', NEW_CRITERION).items.at(-1).definition, /IoU/)
    assert.equal(getDetectionMetricGuide('coco', NEW_CRITERION), getDetectionMetricGuide('coco', OLD_CRITERION))
})

test('no mezcla criterios v1.0 y v1.1 en comparaciones de análisis individuales antiguos sin pipeline', () => {
    const common = {
        task_type: 'detection', dataset_id: 'dataset-a', dataset_revision: 1,
        image_id: 'image-a', model_id: 'yolo26n', attack_type: 'square_detection', parameters: {},
    }
    const left = { ...common, analysis_id: 'a', summary: { ...OLD_CRITERION } }
    const right = { ...common, analysis_id: 'b', summary: { ...NEW_CRITERION } }
    const result = classifyIndividualAnalysisComparison(left, right)
    assert.equal(result.compatible, false)
    assert.equal(result.code, 'different_metrics_schema_version')
    assert.equal(classifyIndividualAnalysisComparison(left, { ...left, analysis_id: 'c', summary: {} }).code, 'identical_configuration')
})

test('resumen A/B de detección conserva el significado del criterio y deja clasificación igual', () => {
    const detection = individualAnalysisComparisonSummary('detection', {
        ...NEW_CRITERION,
        comparison: { attack_success: true, average_confidence_difference: 0.001 },
        original_detections: [{}], adversarial_detections: [{}],
    })
    assert.equal(detection.successLabel, 'Alteración de detecciones')
    assert.match(detection.successExplanation, /IoU/)
    const classification = individualAnalysisComparisonSummary('classification', {
        comparison: { attack_success: true, prediction_changed: true },
    })
    assert.equal(classification.attackSuccess, true)
    assert.equal(classification.successLabel, undefined)
})

test('Square distingue objetivo propio, omisión y presupuesto sin alterar el criterio común', () => {
    const trace = {
        status: 'budget_exhausted', objective_reached: false,
        clean_score: 1, best_score: 0.875, queries: 10, max_queries: 10,
    }
    const info = squareObjectivePresentation(trace)
    assert.equal(info.label, 'Square no alcanzó su objetivo de debilitamiento')
    assert.match(info.detail, /1\.000 → 0\.875/)
    assert.match(info.detail, /10 de 10/)
    assert.match(info.note, /no implica necesariamente perder/)
    assert.equal(squareObjectivePresentation({ ...trace, objective_reached: true }).label, 'Square alcanzó su objetivo de debilitamiento')
    assert.match(squareObjectivePresentation({ ...trace, status: 'skipped_no_references', best_score: null }).label, /no se ejecutó/)
    assert.equal(squareObjectivePresentation(null), null)
})


test('v1.1 explica una confianza caída con cajas conservadas sin proclamar éxito', () => {
    const result = {
        ...NEW_CRITERION,
        comparison: {
            original_detection_count: 2,
            adversarial_detection_count: 2,
            average_original_confidence: 0.9313,
            average_adversarial_confidence: 0.6001,
            attack_success: false,
        },
    }
    const explanation = detectionOutcomeExplanation(result)
    assert.match(explanation, /Las 2 detecciones originales conservan/)
    assert.match(explanation, /cajas sean idénticas/)
    assert.match(explanation, /33\.12 puntos porcentuales/)
    assert.doesNotMatch(explanation, /ataque exitoso/i)
})

test('v1.1 explica pérdidas aun cuando el número de detecciones es idéntico', () => {
    const result = {
        ...NEW_CRITERION,
        original_detection_count: 2,
        adversarial_detection_count: 2,
        attack_success: true,
    }
    const explanation = detectionOutcomeExplanation(result)
    assert.match(explanation, /ha perdido su pareja/)
    assert.match(explanation, /cambio de clase/)
    assert.doesNotMatch(explanation, /detecciones menos/)
})

test('v1.1 distingue nuevas detecciones y casos sin referencias', () => {
    const withExtras = {
        ...NEW_CRITERION,
        original_detection_count: 39,
        adversarial_detection_count: 41,
        attack_success: false,
    }
    assert.match(detectionOutcomeExplanation(withExtras), /2 detecciones adicionales/)
    assert.match(detectionOutcomeExplanation({
        ...NEW_CRITERION,
        original_detection_count: 0,
        adversarial_detection_count: 0,
        attack_success: false,
    }), /No había detecciones originales/)
})

test('resultados históricos siguen con la descripción v1.0', () => {
    const old = {
        ...OLD_CRITERION,
        comparison: {
            original_detection_count: 1,
            adversarial_detection_count: 1,
            average_original_confidence: 0.90,
            average_adversarial_confidence: 0.899,
            attack_success: true,
        },
    }
    assert.match(detectionOutcomeExplanation(old), /histórico v1.0/)
    assert.doesNotMatch(detectionOutcomeExplanation(old), /cajas sean idénticas/)
})

test('Square explica una caída fuerte con un umbral distinto del éxito común', () => {
    const trace = {
        status: 'completed', objective_reached: true,
        clean_score: 1, best_score: 0.609, queries: 2, max_queries: 50,
    }
    const info = squareObjectivePresentation(trace)
    assert.match(info.label, /objetivo de debilitamiento/)
    assert.match(info.detail, /39\.1 % menos/)
    assert.match(info.detail, /2 de 50/)
    assert.match(info.note, /0,80/)
    assert.match(info.note, /IoU ≥ 0,50/)
})
