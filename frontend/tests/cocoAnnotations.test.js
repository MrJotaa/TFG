import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

import {
    canSubmitCocoImport,
    formatCocoImportError,
    getAnnotationStatusLabel,
    getCoverageAfterCocoImport,
    shouldConfirmCocoReplacement,
} from '../src/utils/cocoAnnotations.js'

let viteServer
let CoverageSummary

before(async () => {
    viteServer = await createServer({
        appType: 'custom',
        server: { middlewareMode: true },
    })
    const module = await viteServer.ssrLoadModule(
        '/src/components/CocoAnnotationPanel.jsx',
    )
    CoverageSummary = module.CoverageSummary
})

after(async () => {
    await viteServer?.close()
})

test('coverage summary renders counts and category distribution', () => {
    const html = renderToStaticMarkup(
        createElement(CoverageSummary, {
            coverage: {
                coverage_percentage: 66.67,
                reviewed_images: 2,
                unreviewed_images: 1,
                reviewed_empty_images: 1,
                reviewed_with_objects_images: 1,
                total_objects: 3,
                total_categories: 1,
                objects_by_category: [
                    {
                        category_id: 'category-1',
                        name: 'cat',
                        object_count: 3,
                    },
                ],
            },
        }),
    )

    assert.match(html, /66.67%/)
    assert.match(html, /Imágenes revisadas/)
    assert.match(html, /Revisadas sin objetos/)
    assert.match(html, /Revisadas con objetos/)
    assert.match(html, /cat/)
    assert.match(html, />3</)
})

test('annotation review states have stable public labels', () => {
    assert.equal(getAnnotationStatusLabel('unreviewed'), 'Sin revisar')
    assert.equal(
        getAnnotationStatusLabel('reviewed_empty'),
        'Revisada sin objetos',
    )
    assert.equal(
        getAnnotationStatusLabel('reviewed_with_objects'),
        'Revisada con objetos',
    )
})

test('existing snapshots require replacement confirmation', () => {
    assert.equal(
        shouldConfirmCocoReplacement({ has_annotation_snapshot: true }),
        true,
    )
    assert.equal(
        shouldConfirmCocoReplacement({ has_annotation_snapshot: false }),
        false,
    )
})

test('duplicate submissions are blocked while an import is running', () => {
    const file = { name: 'annotations.json' }

    assert.equal(canSubmitCocoImport({ file, importing: false }), true)
    assert.equal(canSubmitCocoImport({ file, importing: true }), false)
    assert.equal(canSubmitCocoImport({ file: null, importing: false }), false)
})

test('successful imports replace the visible coverage with the API result', () => {
    const coverage = {
        coverage_percentage: 100,
        reviewed_images: 2,
    }

    assert.equal(
        getCoverageAfterCocoImport({ coverage }),
        coverage,
    )
})

test('structured validation errors expose actionable context and rollback', () => {
    const message = formatCocoImportError({
        message: 'fallback',
        details: {
            message: 'La caja delimitadora está fuera de los límites de la imagen de referencia',
            section: 'annotations',
            field: 'bbox',
            coco_id: 42,
            image_name: 'image.png',
            snapshot_applied: false,
        },
    })

    assert.match(message, /caja delimitadora/i)
    assert.match(message, /sección: annotations/)
    assert.match(message, /campo: bbox/)
    assert.match(message, /ID COCO: 42/)
    assert.match(message, /imagen: image.png/)
    assert.match(message, /No se aplicó ningún cambio/)
})
