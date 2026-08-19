import assert from 'node:assert/strict'
import test from 'node:test'

import { streamEvaluationRequest } from '../src/api/evaluationStream.js'

function streamResponse(chunks) {
    const encoder = new TextEncoder()
    return {
        ok: true,
        status: 200,
        body: new ReadableStream({
            start(controller) {
                for (const chunk of chunks) {
                    controller.enqueue(encoder.encode(chunk))
                }
                controller.close()
            },
        }),
    }
}

globalThis.localStorage = {
    getItem() {
        return null
    },
}

test('stream evaluation parses progress across chunk boundaries', async () => {
    const originalFetch = globalThis.fetch
    const progress = []
    globalThis.fetch = async () =>
        streamResponse([
            '{"type":"progress","processed":0,"total":2,"phase":"images"}\n',
            '{"type":"progress","processed":1,',
            '"total":2,"phase":"images"}\n',
            '{"type":"result","data":{"result_id":"evaluation-1"}}\n',
        ])

    try {
        const result = await streamEvaluationRequest(
            '/evaluate/stream',
            { dataset_id: 'dataset-1' },
            { onProgress: (event) => progress.push(event) },
        )

        assert.deepEqual(progress, [
            { processed: 0, total: 2, phase: 'images' },
            { processed: 1, total: 2, phase: 'images' },
        ])
        assert.equal(result.result_id, 'evaluation-1')
    } finally {
        globalThis.fetch = originalFetch
    }
})

test('stream evaluation propagates streamed application errors', async () => {
    const originalFetch = globalThis.fetch
    globalThis.fetch = async () =>
        streamResponse([
            '{"type":"error","status":400,"message":"Invalid dataset"}\n',
        ])

    try {
        await assert.rejects(
            () =>
                streamEvaluationRequest('/evaluate/stream', {
                    dataset_id: 'dataset-1',
                }),
            (error) => {
                assert.equal(error.message, 'Invalid dataset')
                assert.equal(error.status, 400)
                return true
            },
        )
    } finally {
        globalThis.fetch = originalFetch
    }
})


test('stream evaluation exposes cancellation as AbortError', async () => {
    const originalFetch = globalThis.fetch
    globalThis.fetch = async () => streamResponse(['{"type":"cancelled"}\n'])

    try {
        await assert.rejects(
            () => streamEvaluationRequest('/evaluate/stream', { dataset_id: 'dataset-1' }),
            (error) => {
                assert.equal(error.name, 'AbortError')
                return true
            },
        )
    } finally {
        globalThis.fetch = originalFetch
    }
})
