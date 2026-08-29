import assert from 'node:assert/strict'
import test from 'node:test'

import { filterVisibleImages } from '../src/utils/imageGallery.js'

test('image gallery keeps all images when no filter is provided', () => {
    const images = [{ id: 'a' }, { id: 'b' }]
    assert.equal(filterVisibleImages(images, null), images)
})

test('image gallery filter receives only the image argument', () => {
    const images = [{ id: 'a' }, { id: 'b' }]
    const received = []
    const filtered = filterVisibleImages(images, (...args) => {
        received.push(args)
        return args[0].id === 'b'
    })

    assert.deepEqual(filtered, [{ id: 'b' }])
    assert.deepEqual(received, [[images[0]], [images[1]]])
})
