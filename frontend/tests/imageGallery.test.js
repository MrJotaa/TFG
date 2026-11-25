import assert from 'node:assert/strict'
import test from 'node:test'

import { filterVisibleImages, resolveImageSelection } from '../src/utils/imageGallery.js'
import { getExperimentReadiness } from '../src/utils/dashboard.js'

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


test('clicking an unselected image selects it', () => {
    const image = { id: 'a', filename: 'a.jpg' }
    assert.equal(resolveImageSelection(null, image), image)
})

test('clicking the selected image again clears selection by identifier', () => {
    const image = { id: 'a', filename: 'a.jpg' }
    assert.equal(resolveImageSelection({ ...image }, image), null)
})

test('clicking a different image replaces the current selection', () => {
    const image = { id: 'b' }
    assert.equal(resolveImageSelection({ id: 'a' }, image), image)
})

test('deselecting and reselecting an image preserves dataset evaluation readiness', () => {
    const image = { id: 'a' }
    const context = { selectedDataset: { id: 'dataset-1' }, activeModel: { id: 'resnet18' } }
    const deselected = resolveImageSelection(image, image)
    const readiness = getExperimentReadiness({ ...context, selectedImage: deselected })
    assert.equal(readiness.canPredict, false)
    assert.equal(readiness.canAnalyze, false)
    assert.equal(readiness.canEvaluate, true)

    const reselected = resolveImageSelection(deselected, image)
    const restored = getExperimentReadiness({ ...context, selectedImage: reselected })
    assert.equal(restored.canPredict, true)
    assert.equal(restored.canAnalyze, true)
    assert.equal(restored.canEvaluate, true)
})

test('locked image selection ignores clicks on the selected and other images', () => {
    const selected = { id: 'a' }
    assert.equal(resolveImageSelection(selected, { id: 'a' }, true), selected)
    assert.equal(resolveImageSelection(selected, { id: 'b' }, true), selected)
    assert.equal(resolveImageSelection(null, { id: 'b' }, true), null)
})
