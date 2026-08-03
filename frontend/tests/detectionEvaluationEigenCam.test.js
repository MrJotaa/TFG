import assert from 'node:assert/strict'
import test from 'node:test'

import { generateDetectionEvaluationEigenCam } from '../src/api/detectionEvaluation.js'
import {
    canRequestDetectionEigenCam,
    detectionEigenCamReducer,
    getDetectionEigenCamEntry,
    getDetectionEigenCamErrorMessage,
    getDetectionEigenCamKey,
    isDetectionEigenCamResponse,
} from '../src/utils/detectionEvaluationEigenCamState.js'


test('detection evaluation EigenCAM sends one authenticated POST', async () => {
    const requests = []
    const responseBody = {
        evaluation_id: 'detection-evaluation-1',
        image_id: 'image-1',
        explanation_type: 'eigencam',
        original_eigencam_image_url: '/results/original.png',
        adversarial_eigencam_image_url: '/results/adversarial.png',
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

    const response = await generateDetectionEvaluationEigenCam(
        'detection-evaluation-1',
        'image-1',
    )

    assert.deepEqual(response, responseBody)
    assert.equal(requests.length, 1)
    assert.equal(
        requests[0].url,
        'http://127.0.0.1:8000/detect/evaluations/detection-evaluation-1/images/image-1/eigencam',
    )
    assert.equal(requests[0].options.method, 'POST')
    assert.equal(requests[0].options.body, undefined)
    assert.equal(requests[0].options.headers.Authorization, 'Bearer token-1')
})


test('EigenCAM state is isolated and errors permit retry', () => {
    const firstKey = getDetectionEigenCamKey('evaluation-1', 'image-1')
    const secondKey = getDetectionEigenCamKey('evaluation-1', 'image-2')
    let state = {}

    state = detectionEigenCamReducer(state, { type: 'loading', key: firstKey })
    assert.equal(getDetectionEigenCamEntry(state, firstKey).status, 'loading')
    assert.equal(getDetectionEigenCamEntry(state, secondKey).status, 'idle')
    assert.equal(canRequestDetectionEigenCam(state[firstKey]), false)

    state = detectionEigenCamReducer(state, {
        type: 'error',
        key: firstKey,
        error: 'failed',
    })
    assert.equal(canRequestDetectionEigenCam(state[firstKey]), true)
    assert.equal(canRequestDetectionEigenCam(state[firstKey], true), false)

    state = detectionEigenCamReducer(state, { type: 'reset' })
    assert.equal(getDetectionEigenCamEntry(state, firstKey).status, 'idle')
})


test('EigenCAM response requires matching identity, type and both images', () => {
    const validResponse = {
        evaluation_id: 'evaluation-1',
        image_id: 'image-1',
        explanation_type: 'eigencam',
        original_eigencam_image_url: '/original.png',
        adversarial_eigencam_image_url: '/adversarial.png',
    }

    assert.equal(
        isDetectionEigenCamResponse(validResponse, 'evaluation-1', 'image-1'),
        true,
    )
    assert.equal(
        isDetectionEigenCamResponse(validResponse, 'evaluation-2', 'image-1'),
        false,
    )
    assert.equal(
        isDetectionEigenCamResponse(
            { ...validResponse, explanation_type: 'gradcam' },
            'evaluation-1',
            'image-1',
        ),
        false,
    )
})


test('controlled EigenCAM errors have understandable messages', () => {
    for (const status of [400, 401, 403, 404, 410, 500]) {
        const message = getDetectionEigenCamErrorMessage({ status })
        assert.equal(typeof message, 'string')
        assert.ok(message.length > 20)
    }

    assert.match(
        getDetectionEigenCamErrorMessage(new TypeError('network failed')),
        /conectar con el servidor/,
    )
    assert.match(
        getDetectionEigenCamErrorMessage({ status: 410 }),
        /Repite la evaluación/,
    )
})
