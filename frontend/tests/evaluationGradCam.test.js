import assert from 'node:assert/strict'
import test from 'node:test'

import { generateEvaluationGradCam } from '../src/api/evaluation.js'
import { resolveApiUrl } from '../src/api/client.js'
import {
    canRequestEvaluationGradCam,
    evaluationGradCamReducer,
    getEvaluationGradCamEntry,
    getEvaluationGradCamErrorMessage,
    getEvaluationGradCamKey,
    isEvaluationGradCamResponse,
} from '../src/utils/evaluationGradCamState.js'


test('evaluation Grad-CAM client sends one authenticated POST without body', async () => {
    const requests = []
    const responseBody = {
        evaluation_id: 'evaluation-1',
        image_id: 'image-1',
        original_gradcam_image_url: '/results/evaluation-1/original.png',
        adversarial_gradcam_image_url: '/results/evaluation-1/adversarial.png',
    }
    globalThis.localStorage = {
        getItem: (key) => key === 'auth_token' ? 'token-1' : null,
    }
    globalThis.fetch = async (url, options) => {
        requests.push({ url, options })
        return {
            ok: true,
            status: 200,
            json: async () => responseBody,
        }
    }

    const response = await generateEvaluationGradCam(
        'evaluation-1',
        'image-1',
    )

    assert.deepEqual(response, responseBody)
    assert.equal(requests.length, 1)
    assert.equal(
        requests[0].url,
        'http://127.0.0.1:8000/evaluations/evaluation-1/images/image-1/gradcam',
    )
    assert.equal(requests[0].options.method, 'POST')
    assert.equal(requests[0].options.body, undefined)
    assert.equal(requests[0].options.headers.Authorization, 'Bearer token-1')
})


test('state is isolated by evaluation and image and resets for a new run', () => {
    const firstKey = getEvaluationGradCamKey('evaluation-1', 'image-1')
    const secondImageKey = getEvaluationGradCamKey('evaluation-1', 'image-2')
    const secondEvaluationKey = getEvaluationGradCamKey(
        'evaluation-2',
        'image-1',
    )
    let state = {}

    state = evaluationGradCamReducer(state, {
        type: 'loading',
        key: firstKey,
    })
    assert.equal(getEvaluationGradCamEntry(state, firstKey).status, 'loading')
    assert.equal(
        getEvaluationGradCamEntry(state, secondImageKey).status,
        'idle',
    )
    assert.equal(
        getEvaluationGradCamEntry(state, secondEvaluationKey).status,
        'idle',
    )

    state = evaluationGradCamReducer(state, {
        type: 'success',
        key: firstKey,
        data: { original_gradcam_image_url: '/original.png' },
    })
    assert.equal(getEvaluationGradCamEntry(state, firstKey).status, 'success')
    assert.equal(canRequestEvaluationGradCam(state[firstKey]), false)

    state = evaluationGradCamReducer(state, { type: 'reset' })
    assert.equal(getEvaluationGradCamEntry(state, firstKey).status, 'idle')
})


test('loading and success prevent duplicate requests while error permits retry', () => {
    assert.equal(
        canRequestEvaluationGradCam({ status: 'idle' }, false),
        true,
    )
    assert.equal(
        canRequestEvaluationGradCam({ status: 'idle' }, true),
        false,
    )
    assert.equal(
        canRequestEvaluationGradCam({ status: 'loading' }, false),
        false,
    )
    assert.equal(
        canRequestEvaluationGradCam({ status: 'success' }, false),
        false,
    )
    assert.equal(
        canRequestEvaluationGradCam({ status: 'error' }, false),
        true,
    )
})


test('response identity and both URLs are required', () => {
    const validResponse = {
        evaluation_id: 'evaluation-1',
        image_id: 'image-1',
        original_gradcam_image_url: '/original.png',
        adversarial_gradcam_image_url: '/adversarial.png',
    }

    assert.equal(
        isEvaluationGradCamResponse(
            validResponse,
            'evaluation-1',
            'image-1',
        ),
        true,
    )
    assert.equal(
        isEvaluationGradCamResponse(
            validResponse,
            'evaluation-2',
            'image-1',
        ),
        false,
    )
    assert.equal(
        isEvaluationGradCamResponse(
            { ...validResponse, adversarial_gradcam_image_url: '' },
            'evaluation-1',
            'image-1',
        ),
        false,
    )
})


test('relative and absolute image URLs are resolved without duplicate hosts', () => {
    assert.equal(
        resolveApiUrl('/results/evaluation-1/original.png'),
        'http://127.0.0.1:8000/results/evaluation-1/original.png',
    )
    assert.equal(
        resolveApiUrl('https://cdn.example.test/original.png'),
        'https://cdn.example.test/original.png',
    )
    assert.equal(resolveApiUrl(null), null)
})


test('controlled HTTP and network errors have understandable messages', () => {
    for (const status of [400, 401, 403, 404, 410, 500]) {
        const message = getEvaluationGradCamErrorMessage({ status })
        assert.equal(typeof message, 'string')
        assert.ok(message.length > 20)
    }

    assert.match(
        getEvaluationGradCamErrorMessage(new TypeError('network failed')),
        /conectar con el servidor/,
    )
    assert.match(
        getEvaluationGradCamErrorMessage({ status: 410 }),
        /Repite la evaluación/,
    )
})
