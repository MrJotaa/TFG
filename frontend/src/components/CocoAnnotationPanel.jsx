import { useEffect, useRef, useState } from 'react'
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
import DatasetDialog from './DatasetDialog'

function CoverageMetric({ label, value, emphasis = false }) {
    return (
        <div className={`annotation-summary-item${emphasis ? ' emphasis' : ''}`}>
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    )
}

export function CoverageSummary({ coverage }) {
    return (
        <div className="annotation-coverage-modern">
            <div className="annotation-summary-strip">
                <CoverageMetric
                    label="Cobertura"
                    value={`${coverage.coverage_percentage}%`}
                    emphasis
                />
                <CoverageMetric
                    label="Revisadas"
                    value={`${coverage.reviewed_images} / ${coverage.total_images}`}
                />
                <CoverageMetric label="Objetos" value={coverage.total_objects} />
                <CoverageMetric label="Categorías" value={coverage.total_categories} />
            </div>

            <div className="annotation-review-breakdown">
                <span>{coverage.unreviewed_images} sin revisar</span>
                <span>{coverage.reviewed_empty_images} revisadas sin objetos</span>
                <span>{coverage.reviewed_with_objects_images} con objetos</span>
            </div>

            <div className="annotation-category-distribution modern">
                <h4>Categorías</h4>
                {coverage.objects_by_category.length === 0 ? (
                    <p className="muted-text">
                        Todavía no hay categorías disponibles en las anotaciones.
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
        </div>
    )
}

function CocoAnnotationPanel({ selectedDataset, onAnnotationsImported }) {
    const formRef = useRef(null)
    const [coverage, setCoverage] = useState(null)
    const [selectedFile, setSelectedFile] = useState(null)
    const [importing, setImporting] = useState(false)
    const [confirmReplacement, setConfirmReplacement] = useState(false)
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

    async function performImport() {
        if (!canSubmitCocoImport({ file: selectedFile, importing })) return

        try {
            setImporting(true)
            setConfirmReplacement(false)
            setError('')
            setSuccess('')

            const result = await importDatasetCocoAnnotations({
                datasetId: selectedDataset.id,
                file: selectedFile,
            })

            setCoverage(getCoverageAfterCocoImport(result))
            setSelectedFile(null)
            formRef.current?.reset()
            setSuccess(
                result.replaced_existing_snapshot
                    ? 'Anotaciones COCO reemplazadas correctamente.'
                    : 'Anotaciones COCO importadas correctamente.',
            )
            onAnnotationsImported?.(result)
        } catch (requestError) {
            setError(formatCocoImportError(requestError))
        } finally {
            setImporting(false)
        }
    }

    function handleImport(event) {
        event.preventDefault()
        if (!canSubmitCocoImport({ file: selectedFile, importing })) return

        if (shouldConfirmCocoReplacement(visibleCoverage)) {
            setConfirmReplacement(true)
            return
        }

        performImport()
    }

    let importButtonLabel = 'Importar anotaciones'
    if (importing) importButtonLabel = 'Importando...'
    else if (visibleCoverage?.has_annotation_snapshot) importButtonLabel = 'Reemplazar anotaciones'

    return (
        <section className="coco-annotation-panel modern">
            <div className="coco-annotation-heading">
                <div>
                    <span className="dashboard-section-eyebrow">Anotaciones COCO</span>
                    <h3>Detección y cobertura</h3>
                    <p>
                        Importa un JSON COCO para asociar categorías y cajas a las imágenes del conjunto.
                    </p>
                </div>
                {visibleCoverage?.has_annotation_snapshot && (
                    <span className="dataset-status-badge success">COCO activo</span>
                )}
            </div>

            {!visibleCoverage ? (
                <p className="muted-text">Cargando cobertura de anotaciones...</p>
            ) : (
                <CoverageSummary coverage={visibleCoverage} />
            )}

            <form ref={formRef} className="coco-import-form modern" onSubmit={handleImport}>
                <label>
                    <span>Archivo de anotaciones</span>
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
                    disabled={!canSubmitCocoImport({ file: selectedFile, importing })}
                >
{importButtonLabel}
                </button>
            </form>

            <p className="coco-import-note">
                Cada importación sustituye el snapshot anterior de forma atómica; las anotaciones no se mezclan.
            </p>
            {error && <p className="error dataset-inline-feedback">{error}</p>}
            {success && <p className="success-message dataset-inline-feedback">{success}</p>}

            {confirmReplacement && (
                <DatasetDialog
                    title="Reemplazar anotaciones COCO"
                    eyebrow="Confirmación"
                    onClose={() => setConfirmReplacement(false)}
                    busy={importing}
                    danger
                >
                    <div className="dataset-dialog-body">
                        <p>
                            Este conjunto ya tiene un snapshot de anotaciones. La nueva importación lo sustituirá por completo.
                        </p>
                    </div>
                    <div className="dataset-dialog-actions">
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={() => setConfirmReplacement(false)}
                            disabled={importing}
                        >
                            Cancelar
                        </button>
                        <button
                            type="button"
                            className="danger-action"
                            onClick={performImport}
                            disabled={importing}
                        >
                            {importing ? 'Reemplazando...' : 'Reemplazar anotaciones'}
                        </button>
                    </div>
                </DatasetDialog>
            )}
        </section>
    )
}

export default CocoAnnotationPanel
