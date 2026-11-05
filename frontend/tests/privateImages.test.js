import test from 'node:test'
import assert from 'node:assert/strict'

import { fetchPrivateImage, isPrivateImageUrl } from '../src/api/privateImages.js'

const API = 'http://127.0.0.1:8000'

test('only images served by our API are fetched with JWT', () => {
    assert.equal(isPrivateImageUrl('/datasets/123/images/abc'), true)
    assert.equal(isPrivateImageUrl('/results/attack_123/adversarial.png'), true)
    assert.equal(isPrivateImageUrl(`${API}/results/evaluation_1/original.png`), true)
    assert.equal(isPrivateImageUrl('https://other.example/results/abc/image.png'), false)
    assert.equal(isPrivateImageUrl('/auth/me'), false)
    assert.equal(isPrivateImageUrl('/results/123'), false)
    assert.equal(isPrivateImageUrl(null), false)
})

test('private images are fetched with Authorization, not a token in the URL', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    const previousCreate = URL.createObjectURL
    const expectedBlobUrl = 'blob:private-image-test'
    const seen = []
    globalThis.localStorage = { getItem: () => 'test-jwt' }
    URL.createObjectURL = () => expectedBlobUrl
    globalThis.fetch = async (url, options) => {
        seen.push({ url, options })
        return new Response(new Blob(['bytes'], { type: 'image/png' }), { status: 200 })
    }
    try {
        const image = await fetchPrivateImage('/datasets/d/images/i')
        assert.equal(image, expectedBlobUrl)
        assert.equal(seen.length, 1)
        assert.equal(seen[0].url, `${API}/datasets/d/images/i`)
        assert.equal(seen[0].options.headers.Authorization, 'Bearer test-jwt')
        assert.equal(seen[0].options.cache, 'no-store')
        assert.doesNotMatch(seen[0].url, /test-jwt/)
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
        URL.createObjectURL = previousCreate
    }
})

test('a private image with no token is never requested anonymously', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    let requests = 0
    globalThis.localStorage = { getItem: () => null }
    globalThis.fetch = () => { requests += 1 }
    try {
        await assert.rejects(
            fetchPrivateImage('/results/attack_123/adversarial.png'),
            /iniciar sesión/,
        )
        assert.equal(requests, 0)
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
    }
})

test('private media fetch exposes HTTP errors instead of silently returning a blank image', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    globalThis.localStorage = { getItem: () => 'test-jwt' }
    globalThis.fetch = async () => new Response(
        JSON.stringify({ detail: 'El conjunto de datos pertenece a otro usuario' }),
        { status: 403, headers: { 'Content-Type': 'application/json' } },
    )
    try {
        await assert.rejects(
            fetchPrivateImage('/datasets/d/images/i'),
            (error) => error.status === 403 && /otro usuario/.test(error.message),
        )
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
    }
})

test('private media fetch does not accept an HTML or empty API response as an image', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    globalThis.localStorage = { getItem: () => 'test-jwt' }
    globalThis.fetch = async () => new Response('<html>Not an image</html>', {
        status: 200,
        headers: { 'Content-Type': 'text/html' },
    })
    try {
        await assert.rejects(fetchPrivateImage('/results/a/b.png'), /imagen válida/)
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
    }
})

test('a protected image uses the latest session token and propagates cancellation', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    const previousCreate = URL.createObjectURL
    let token = 'first-session-token'
    const requests = []
    const controller = new AbortController()

    globalThis.localStorage = { getItem: () => token }
    URL.createObjectURL = () => 'blob:protected-test-image'
    globalThis.fetch = async (url, options) => {
        requests.push({ url, options })
        return new Response(new Blob(['image-bytes'], { type: 'image/png' }), {
            status: 200,
        })
    }
    try {
        await fetchPrivateImage('/results/attack_1/adversarial.png', {
            signal: controller.signal,
        })
        token = 'second-session-token'
        await fetchPrivateImage('/results/attack_1/adversarial.png')

        assert.equal(requests.length, 2)
        assert.equal(requests[0].options.headers.Authorization, 'Bearer first-session-token')
        assert.equal(requests[1].options.headers.Authorization, 'Bearer second-session-token')
        assert.equal(requests[0].options.signal, controller.signal)
        assert.equal(requests[0].options.cache, 'no-store')
        assert.equal(requests[1].options.cache, 'no-store')
        for (const request of requests) {
            assert.doesNotMatch(request.url, /session-token/)
        }
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
        URL.createObjectURL = previousCreate
    }
})

test('a foreign image URL never receives the session JWT', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    let requests = 0

    globalThis.localStorage = { getItem: () => 'secret-session' }
    globalThis.fetch = async () => { requests += 1 }
    try {
        await assert.rejects(
            fetchPrivateImage('https://other.example/results/attack_1/image.png'),
            /imagen privada/,
        )
        assert.equal(requests, 0)
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
    }
})

test('an empty successful response is not converted into an image', async () => {
    const previousFetch = globalThis.fetch
    const previousStorage = globalThis.localStorage
    globalThis.localStorage = { getItem: () => 'test-jwt' }
    globalThis.fetch = async () => new Response('', {
        status: 200,
        headers: { 'Content-Type': 'image/png' },
    })
    try {
        await assert.rejects(
            fetchPrivateImage('/datasets/d/images/i'),
            /imagen válida/,
        )
    } finally {
        globalThis.fetch = previousFetch
        globalThis.localStorage = previousStorage
    }
})
