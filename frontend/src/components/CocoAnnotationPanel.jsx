import { useEffect, useState } from 'react'
import {
    getDatasetAnnotationCoverage,
    importDatasetCocoAnnotations,
} from '../api/datasets'
import {
    canSubmitCocoImport,
    formatCocoImportError,
    getCoverageAfterCocoImport,
    shouldConfirmCocoReplacement,
} from '../utils/cocoAnnotations'

function CoverageMetric({ label, value }) {
    return (
        <article className="metric-card">
            <span>{label}</span>
            <strong>{value}</strong>
        </article>
    )
}

export function CoverageSummary({ coverage }) {
    return (
        <>
            <div className="metric-grid annotation-coverage-grid">
                <CoverageMetric
                    label="Cobertura"
                    value={`${coverage.coverage_percentage}%`}
                />
                <CoverageMetric
                    label="Imágenes revisadas"
                    value={coverage.reviewed_images}
                />
                <CoverageMetric
                    label="Sin revisar"
                    value={coverage.unreviewed_images}
                />
                <CoverageMetric
                    label="Revisadas sin objetos"
                    value={coverage.reviewed_empty_images}
                />
                <CoverageMetric
                    label="Revisadas con objetos"
                    value={coverage.reviewed_with_objects_images}
                />
                <CoverageMetric label="Objetos" value={coverage.total_objects} />
                <CoverageMetric
                    label="Categorías"
                    value={coverage.total_categories}
                />
            </div>

            <div className="annotation-category-distribution">
                <h5>Objetos por categoría</h5>
                {coverage.objects_by_category.length === 0 ? (
                    <p className="muted-text">
                        El snapshot no contiene categorías.
                    </p>
                ) : (
                    <ul>
                        {coverage.objects_by_category.map((category) => (
                            <li key={category.category_id}>
                                <span>{category.name}</span>
                                <strong>{category.object_count}</strong>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </>
    )
}

function CocoAnnotationPanel({ selectedDataset, onAnnotationsImported }) {
    const [coverage, setCoverage] = useState(null)
    const [selectedFile, setSelectedFile] = useState(null)
    const [importing, setImporting] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState('')

    useEffect(() => {
        let active = true

        getDatasetAnnotationCoverage(selectedDataset.id)
            .then((result) => {
                if (!active) return
                setCoverage(result)
                setError('')
            })
            .catch((requestError) => {
                if (!active) return
                setError(formatCocoImportError(requestError))
            })

        return () => {
            active = false
        }
    }, [selectedDataset.id])

    const visibleCoverage =
        coverage?.dataset_id === selectedDataset.id ? coverage : null

    async function handleImport(event) {
        event.preventDefault()
        const form = event.currentTarget

        if (!canSubmitCocoImport({ file: selectedFile, importing })) {
            return
        }

        if (
            shouldConfirmCocoReplacement(visibleCoverage)
            && !window.confirm(
                'Esta importación reemplazará por completo el snapshot de anotaciones actual. ¿Quieres continuar?',
            )
        ) {
            return
        }

        try {
            setImporting(true)
            setError('')
            setSuccess('')

            const result = await importDatasetCocoAnnotations({
                datasetId: selectedDataset.id,
                file: selectedFile,
            })

            setCoverage(getCoverageAfterCocoImport(result))
            setSelectedFile(null)
            form.reset()
            setSuccess(
                result.replaced_existing_snapshot
                    ? 'Snapshot COCO reemplazado correctamente.'
                    : 'Anotaciones COCO importadas correctamente.',
            )
            onAnnotationsImported?.(result)
        } catch (requestError) {
            setError(formatCocoImportError(requestError))
        } finally {
            setImporting(false)
        }
    }

    return (
        <section className="coco-annotation-panel">
            <div>
                <h4>Anotaciones de detección</h4>
                <p>
                    Importa un archivo COCO JSON. Cada importación es un snapshot
                    completo y reemplaza el anterior; no se mezclan anotaciones.
                </p>
            </div>

            <form className="coco-import-form" onSubmit={handleImport}>
                <label>
                    Archivo COCO JSON
                    <input
                        type="file"
                        accept=".json,application/json"
                        disabled={importing}
                        onChange={(event) => {
                            setSelectedFile(event.target.files?.[0] || null)
                            setError('')
                            setSuccess('')
                        }}
                    />
                </label>

                <button
                    type="submit"
                    className="primary-action"
                    disabled={
                        !canSubmitCocoImport({
                            file: selectedFile,
                            importing,
                        })
                    }
                >
                    {importing
                        ? 'Importando anotaciones...'
                        : visibleCoverage?.has_annotation_snapshot
                          ? 'Reemplazar snapshot COCO'
                          : 'Importar anotaciones COCO'}
                </button>
            </form>

            {error && <p className="error">{error}</p>}
            {success && <p className="success-message">{success}</p>}

            {!visibleCoverage ? (
                <p className="muted-text">Cargando cobertura de anotaciones...</p>
            ) : (
                <CoverageSummary coverage={visibleCoverage} />
            )}
        </section>
    )
}

export default CocoAnnotationPanel
