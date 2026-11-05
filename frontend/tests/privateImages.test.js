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
