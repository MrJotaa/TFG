import assert from 'node:assert/strict'
import { after, afterEach, before, beforeEach, test } from 'node:test'
import { createServer } from 'vite'

let viteServer
let client
let auth
let datasets
let models
let attacks
let predict
let attack
let detect
let explainability
let evaluation
let detectionEvaluation
let evaluationHistory
let individualAnalysisHistory
let originalFetch
let storage

before(async () => {
    storage = new Map()
    globalThis.localStorage = {
        getItem(key) {
            return storage.has(key) ? storage.get(key) : null
        },
        setItem(key, value) {
            storage.set(key, String(value))
        },
        removeItem(key) {
            storage.delete(key)
        },
        clear() {
            storage.clear()
        },
    }

    viteServer = await createServer({
        appType: 'custom',
        server: { middlewareMode: true },
    })

    ;[
        client,
        auth,
        datasets,
        models,
        attacks,
        predict,
        attack,
        detect,
        explainability,
        evaluation,
        detectionEvaluation,
        evaluationHistory,
        individualAnalysisHistory,
    ] = await Promise.all([
        viteServer.ssrLoadModule('/src/api/client.js'),
        viteServer.ssrLoadModule('/src/api/auth.js'),
        viteServer.ssrLoadModule('/src/api/datasets.js'),
        viteServer.ssrLoadModule('/src/api/models.js'),
        viteServer.ssrLoadModule('/src/api/attacks.js'),
        viteServer.ssrLoadModule('/src/api/predict.js'),
        viteServer.ssrLoadModule('/src/api/attack.js'),
        viteServer.ssrLoadModule('/src/api/detect.js'),
        viteServer.ssrLoadModule('/src/api/explainability.js'),
        viteServer.ssrLoadModule('/src/api/evaluation.js'),
        viteServer.ssrLoadModule('/src/api/detectionEvaluation.js'),
        viteServer.ssrLoadModule('/src/api/evaluationHistory.js'),
        viteServer.ssrLoadModule('/src/api/individualAnalysisHistory.js'),
    ])
})

beforeEach(() => {
    storage.clear()
    originalFetch = globalThis.fetch
})

afterEach(() => {
    globalThis.fetch = originalFetch
})

after(async () => {
    await viteServer?.close()
})

function jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' },
    })
}

function streamResponse(lines) {
    const encoder = new TextEncoder()
    return new Response(
        new ReadableStream({
            start(controller) {
                for (const line of lines) {
                    controller.enqueue(encoder.encode(line))
                }
                controller.close()
            },
        }),
        { status: 200 },
    )
}

function captureFetch(responseFactory = () => jsonResponse({ ok: true })) {
    const calls = []
    globalThis.fetch = async (url, options = {}) => {
        calls.push({ url, options })
        return responseFactory(url, options, calls.length - 1)
    }
    return calls
}

function requestBody(call) {
    return JSON.parse(call.options.body)
}

test('authenticated session sends bearer token and logout removes it', async () => {
    client.setStoredToken('session-token')
    const calls = captureFetch(() =>
        jsonResponse({ user: { id: 'user-1', email: 'analyst@example.test' } }),
    )

    await auth.getCurrentUser()

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/auth/me')
    assert.equal(calls[0].options.headers.Authorization, 'Bearer session-token')
    assert.equal(client.getStoredToken(), 'session-token')

    client.removeStoredToken()
    assert.equal(client.getStoredToken(), null)
})

test('login and API errors preserve the backend contract', async () => {
    const calls = captureFetch((_url, _options, index) => {
        if (index === 0) {
            return jsonResponse({ access_token: 'token', user: { id: 'user-1' } })
        }
        return jsonResponse(
            { detail: { message: 'Dataset no disponible', code: 'dataset_missing' } },
            404,
        )
    })

    await auth.loginUser({
        email: 'analyst@example.test',
        password: 'secret',
    })

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/auth/login')
    assert.equal(calls[0].options.method, 'POST')
    assert.deepEqual(requestBody(calls[0]), {
        email: 'analyst@example.test',
        password: 'secret',
    })

    await assert.rejects(
        () => client.apiRequest('/datasets/missing'),
        (error) => {
            assert.equal(error.message, 'Dataset no disponible')
            assert.equal(error.status, 404)
            assert.deepEqual(error.details, {
                message: 'Dataset no disponible',
                code: 'dataset_missing',
            })
            return true
        },
    )
})

test('dataset upload keeps multipart headers browser-managed', async () => {
    client.setStoredToken('session-token')
    const calls = captureFetch(() => jsonResponse({ id: 'dataset-1' }))
    const image = new File(['image-bytes'], 'sample.png', { type: 'image/png' })

    await datasets.uploadDataset({
        userId: 'user-1',
        name: 'COCO sample',
        taskType: 'detection',
        files: [image],
    })

    const call = calls[0]
    assert.equal(call.url, 'http://127.0.0.1:8000/datasets')
    assert.equal(call.options.method, 'POST')
    assert.equal(call.options.headers.Authorization, 'Bearer session-token')
    assert.equal(call.options.headers['Content-Type'], undefined)
    assert.ok(call.options.body instanceof FormData)
    assert.equal(call.options.body.get('user_id'), 'user-1')
    assert.equal(call.options.body.get('name'), 'COCO sample')
    assert.equal(call.options.body.get('task_type'), 'detection')
    assert.equal(call.options.body.get('files').name, 'sample.png')
})


test('dataset metadata update uses the dataset PATCH contract', async () => {
    client.setStoredToken('session-token')
    const calls = captureFetch(() =>
        jsonResponse({ id: 'dataset-1', name: 'Renamed', task_type: 'classification' }),
    )

    await datasets.updateDataset({
        datasetId: 'dataset-1',
        name: 'Renamed',
        taskType: 'classification',
    })

    const call = calls[0]
    assert.equal(call.url, 'http://127.0.0.1:8000/datasets/dataset-1')
    assert.equal(call.options.method, 'PATCH')
    assert.deepEqual(requestBody(call), {
        name: 'Renamed',
        task_type: 'classification',
    })
})

test('attack catalog is retrieved from the backend source of truth', async () => {
    const calls = captureFetch(() => jsonResponse([]))

    await attacks.getAttacks()

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/attacks')
})

test('model selection and clean inference use the expected classification and detection endpoints', async () => {
    const calls = captureFetch((_url, _options, index) =>
        jsonResponse(index === 0 ? { id: 'yolo26n', task_type: 'detection' } : {}),
    )

    await models.selectModel('yolo26n')
    await predict.predictImage({ datasetId: 'dataset-1', imageId: 'image-1' })
    await detect.detectImage({ datasetId: 'dataset-1', imageId: 'image-1' })

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/models/yolo26n/select')
    assert.equal(calls[0].options.method, 'POST')

    assert.equal(calls[1].url, 'http://127.0.0.1:8000/predict')
    assert.deepEqual(requestBody(calls[1]), {
        dataset_id: 'dataset-1',
        image_id: 'image-1',
    })

    assert.equal(calls[2].url, 'http://127.0.0.1:8000/detect')
    assert.deepEqual(requestBody(calls[2]), {
        dataset_id: 'dataset-1',
        image_id: 'image-1',
    })
})

test('individual adversarial flows preserve attack parameters and explainability routes', async () => {
    const calls = captureFetch(() => jsonResponse({ ok: true }))

    await attack.applyAttack({
        datasetId: 'dataset-1',
        imageId: 'image-1',
        attackType: 'pgd',
        parameters: { epsilon: 0.03, alpha: 0.01, steps: 10 },
    })
    await explainability.generateGradCam({
        datasetId: 'dataset-1',
        imageId: 'image-1',
    })
    await detect.attackDetectionImage({
        datasetId: 'dataset-1',
        imageId: 'image-1',
        attackType: 'yolo_confidence',
        parameters: { epsilon: 0.03, alpha: 0.01, steps: 10 },
    })
    await explainability.generateDetectionExplainability({
        datasetId: 'dataset-1',
        imageId: 'image-1',
    })

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/attack')
    assert.deepEqual(requestBody(calls[0]).parameters, {
        epsilon: 0.03,
        alpha: 0.01,
        steps: 10,
    })
    assert.equal(calls[1].url, 'http://127.0.0.1:8000/explain/gradcam')

    assert.equal(calls[2].url, 'http://127.0.0.1:8000/detect/attack')
    assert.deepEqual(requestBody(calls[2]).parameters, {
        epsilon: 0.03,
        alpha: 0.01,
        steps: 10,
    })
    assert.equal(calls[3].url, 'http://127.0.0.1:8000/explain/detection')
})

test('classification dataset evaluation streams real progress and final result', async () => {
    const progress = []
    const calls = captureFetch(() =>
        streamResponse([
            '{"type":"progress","processed":0,"total":2,"phase":"images"}\n',
            '{"type":"progress","processed":2,"total":2,"phase":"metrics"}\n',
            '{"type":"result","data":{"result_id":"classification-eval"}}\n',
        ]),
    )

    const result = await evaluation.evaluateDataset({
        datasetId: 'dataset-1',
        attackType: 'pgd',
        parameters: { epsilon: 0.03, alpha: 0.01, steps: 10 },
        forceRecompute: true,
        onProgress: (event) => progress.push(event),
    })

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/evaluate/stream')
    assert.deepEqual(requestBody(calls[0]), {
        dataset_id: 'dataset-1',
        attack_type: 'pgd',
        parameters: {
            epsilon: 0.03,
            alpha: 0.01,
            steps: 10,
        },
        force_recompute: true,
    })
    assert.deepEqual(progress, [
        { processed: 0, total: 2, phase: 'images' },
        { processed: 2, total: 2, phase: 'metrics' },
    ])
    assert.equal(result.result_id, 'classification-eval')
})

test('detection evaluation keeps supervised configuration in the end-to-end flow', async () => {
    const configuration = {
        confidence_threshold: 0.25,
        iou_threshold: 0.5,
        include_object_level_details: true,
    }
    const calls = captureFetch(() =>
        streamResponse([
            '{"type":"result","data":{"result_id":"detection-eval"}}\n',
        ]),
    )

    const result = await detectionEvaluation.evaluateDetectionDataset({
        datasetId: 'dataset-2',
        attackType: 'yolo_confidence',
        parameters: { epsilon: 0.03, alpha: 0.01, steps: 10 },
        supervisedConfiguration: configuration,
        forceRecompute: false,
    })

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/detect/evaluate/stream')
    assert.deepEqual(requestBody(calls[0]), {
        dataset_id: 'dataset-2',
        attack_type: 'yolo_confidence',
        parameters: {
            epsilon: 0.03,
            alpha: 0.01,
            steps: 10,
        },
        supervised_configuration: configuration,
        force_recompute: false,
    })
    assert.equal(result.result_id, 'detection-eval')
})

test('evaluation compatibility and recovery preserve task-specific identity', async () => {
    const calls = captureFetch((_url, _options, index) =>
        jsonResponse(index === 0 ? { match: null } : { task_type: 'detection' }),
    )
    const supervisedConfiguration = {
        confidence_threshold: 0.25,
        iou_threshold: 0.5,
        include_object_level_details: false,
    }

    await evaluationHistory.findCompatibleEvaluation({
        taskType: 'detection',
        datasetId: 'dataset-2',
        attackType: 'yolo_confidence',
        parameters: { epsilon: 0.03, alpha: 0.01, steps: 10 },
        supervisedConfiguration,
    })
    await evaluationHistory.getStoredEvaluation('evaluation/with spaces')

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/evaluations/compatibility')
    assert.deepEqual(requestBody(calls[0]), {
        task_type: 'detection',
        dataset_id: 'dataset-2',
        attack_type: 'yolo_confidence',
        parameters: {
            epsilon: 0.03,
            alpha: 0.01,
            steps: 10,
        },
        supervised_configuration: supervisedConfiguration,
    })
    assert.equal(
        calls[1].url,
        'http://127.0.0.1:8000/evaluations/evaluation%2Fwith%20spaces',
    )
})


test('individual analysis persistence and duplicate lookup use stable history contracts', async () => {
    const calls = captureFetch(() => jsonResponse({ match: null }))

    await individualAnalysisHistory.findCompatibleIndividualAnalysis({
        taskType: 'classification',
        datasetId: 'dataset-1',
        imageId: 'image-1',
        attackType: 'fgsm',
        parameters: { epsilon: 0.03 },
    })
    await individualAnalysisHistory.saveIndividualAnalysis({
        taskType: 'classification',
        resultId: 'attack-result-1',
    })

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/individual-analyses/compatibility')
    assert.deepEqual(requestBody(calls[0]), {
        task_type: 'classification',
        dataset_id: 'dataset-1',
        image_id: 'image-1',
        attack_type: 'fgsm',
        parameters: { epsilon: 0.03 },
    })
    assert.equal(calls[1].url, 'http://127.0.0.1:8000/individual-analyses')
    assert.deepEqual(requestBody(calls[1]), {
        task_type: 'classification',
        result_id: 'attack-result-1',
    })
})

test('evaluation history exposes preserve and delete operations', async () => {
    const calls = captureFetch(() => jsonResponse({ preserved: true }))

    await evaluationHistory.setEvaluationPreserved('evaluation-1', true)
    await evaluationHistory.deleteEvaluation('evaluation-1')

    assert.equal(calls[0].url, 'http://127.0.0.1:8000/evaluations/evaluation-1/preservation')
    assert.equal(calls[0].options.method, 'PATCH')
    assert.deepEqual(requestBody(calls[0]), { preserved: true })
    assert.equal(calls[1].url, 'http://127.0.0.1:8000/evaluations/evaluation-1')
    assert.equal(calls[1].options.method, 'DELETE')
})
