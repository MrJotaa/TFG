import { useMemo, useState } from 'react'
import { formatInteger, formatRatio } from '../../utils/metricFormatting'

function calculateMatrixStats(matrix) {
    let total = 0
    let correct = 0

    matrix.forEach((row, rowIndex) => {
        row.forEach((value, columnIndex) => {
            const numericValue = Number(value) || 0
            total += numericValue

            if (rowIndex === columnIndex) {
                correct += numericValue
            }
        })
    })

    return {
        total,
        correct,
        errors: Math.max(0, total - correct),
        accuracy: total > 0 ? correct / total : 0,
    }
}

function getCellStyle(value, maximum, mode, variant) {
    const numericValue = Number(value) || 0
    const intensity = maximum > 0 ? numericValue / maximum : 0
    const alpha = numericValue > 0 ? 0.08 + intensity * 0.72 : 0.035
    const rgb = variant === 'original' ? '37, 99, 235' : '220, 38, 38'

    return {
        backgroundColor: `rgba(${rgb}, ${alpha})`,
        color: intensity > 0.62 ? '#ffffff' : '#0f172a',
        opacity: mode === 'normalized' && numericValue === 0 ? 0.68 : 1,
    }
}

function formatCellValue(value, mode) {
    return mode === 'normalized' ? formatRatio(value, 0) : formatInteger(value)
}

function EvaluationConfusionMatrices({ evaluation }) {
    const [variant, setVariant] = useState('original')
    const [mode, setMode] = useState('counts')
    const [viewMode, setViewMode] = useState('compact')

    const matrices = evaluation?.advanced_metrics?.supervised?.confusion_matrices

    const matrix = useMemo(() => {
        if (!matrices) {
            return []
        }

        if (mode === 'normalized') {
            return variant === 'original'
                ? matrices.original_normalized
                : matrices.adversarial_normalized
        }

        return variant === 'original' ? matrices.original : matrices.adversarial
    }, [matrices, mode, variant])

    const countMatrix = useMemo(() => {
        if (!matrices) {
            return []
        }

        return variant === 'original' ? matrices.original : matrices.adversarial
    }, [matrices, variant])

    const visibleRows = useMemo(
        () =>
            matrix
                .map((values, sourceIndex) => ({ sourceIndex, values }))
                .filter(({ sourceIndex }) => {
                    if (viewMode === 'complete') {
                        return true
                    }

                    return (countMatrix[sourceIndex] ?? []).some(
                        (value) => (Number(value) || 0) > 0,
                    )
                }),
        [countMatrix, matrix, viewMode],
    )

    const stats = useMemo(() => calculateMatrixStats(countMatrix), [countMatrix])
    const maximum = useMemo(
        () =>
            Math.max(
                0,
                ...visibleRows.flatMap(({ values }) =>
                    values.map((value) => Number(value) || 0),
                ),
            ),
        [visibleRows],
    )

    if (!matrices?.labels?.length || !matrix.length) {
        return (
            <div className="evaluation-empty-panel">
                <span className="evaluation-eyebrow">Matriz de confusión</span>
                <h3>No hay matrices disponibles</h3>
                <p>Se necesitan etiquetas resueltas para construir esta visualización.</p>
            </div>
        )
    }

    const labels = matrices.labels
    const selectedName = variant === 'original' ? 'original' : 'adversarial'
    const hiddenRowCount = Math.max(0, labels.length - visibleRows.length)

    return (
        <div className="evaluation-tab-content confusion-matrix-tab">
            <section className="evaluation-surface confusion-matrix-header">
                <div>
                    <span className="evaluation-eyebrow">Patrones de error</span>
                    <h3>Matriz de confusión {selectedName}</h3>
                    <p>
                        Las filas representan la clase real y las columnas la clase
                        predicha. La diagonal contiene los aciertos.
                    </p>
                </div>

                <div className="confusion-matrix-controls">
                    <div
                        className="evaluation-segmented-control confusion-variant-control"
                        role="group"
                        aria-label="Predicciones mostradas"
                    >
                        <button
                            type="button"
                            className={variant === 'original' ? 'active' : ''}
                            data-variant="original"
                            aria-pressed={variant === 'original'}
                            onClick={() => setVariant('original')}
                        >
                            Original
                        </button>
                        <button
                            type="button"
                            className={variant === 'adversarial' ? 'active' : ''}
                            data-variant="adversarial"
                            aria-pressed={variant === 'adversarial'}
                            onClick={() => setVariant('adversarial')}
                        >
                            Adversarial
                        </button>
                    </div>

                    <div
                        className="evaluation-segmented-control"
                        role="group"
                        aria-label="Representación de la matriz"
                    >
                        <button
                            type="button"
                            className={mode === 'counts' ? 'active' : ''}
                            aria-pressed={mode === 'counts'}
                            onClick={() => setMode('counts')}
                        >
                            Conteos
                        </button>
                        <button
                            type="button"
                            className={mode === 'normalized' ? 'active' : ''}
                            aria-pressed={mode === 'normalized'}
                            onClick={() => setMode('normalized')}
                        >
                            Normalizada
                        </button>
                    </div>

                    <div
                        className="evaluation-segmented-control"
                        role="group"
                        aria-label="Densidad de la matriz"
                    >
                        <button
                            type="button"
                            className={viewMode === 'compact' ? 'active' : ''}
                            aria-pressed={viewMode === 'compact'}
                            onClick={() => setViewMode('compact')}
                        >
                            Compacta
                        </button>
                        <button
                            type="button"
                            className={viewMode === 'complete' ? 'active' : ''}
                            aria-pressed={viewMode === 'complete'}
                            onClick={() => setViewMode('complete')}
                        >
                            Completa
                        </button>
                    </div>
                </div>
            </section>

            <section className="confusion-matrix-stat-grid" aria-label="Resumen de matriz">
                <article>
                    <span>Muestras</span>
                    <strong>{formatInteger(stats.total)}</strong>
                </article>
                <article>
                    <span>Aciertos</span>
                    <strong>{formatInteger(stats.correct)}</strong>
                </article>
                <article>
                    <span>Errores</span>
                    <strong>{formatInteger(stats.errors)}</strong>
                </article>
                <article>
                    <span>Exactitud</span>
                    <strong>{formatRatio(stats.accuracy)}</strong>
                </article>
            </section>

            <section className="evaluation-surface confusion-matrix-surface">
                <div className="confusion-matrix-viewport-header">
                    <div>
                        <strong>
                            {viewMode === 'compact' ? 'Vista compacta' : 'Vista completa'}
                        </strong>
                        <span>
                            {formatInteger(visibleRows.length)} de{' '}
                            {formatInteger(labels.length)} filas visibles
                            {viewMode === 'compact' && hiddenRowCount > 0
                                ? ` · ${formatInteger(hiddenRowCount)} filas sin muestras ocultas`
                                : ''}
                        </span>
                    </div>
                    <p>Desplázate horizontal y verticalmente para explorar la matriz.</p>
                </div>

                <div className="confusion-matrix-axis-label top">Clase predicha</div>
                <div className="confusion-matrix-layout">
                    <div className="confusion-matrix-axis-label side">Clase real</div>

                    <div
                        className="confusion-matrix-scroll"
                        role="region"
                        aria-label={`Matriz de confusión ${selectedName}`}
                        tabIndex="0"
                    >
                        <table
                            className={`confusion-matrix-table ${
                                viewMode === 'compact' ? 'is-compact' : 'is-complete'
                            }`}
                        >
                            <caption>
                                Matriz de confusión {selectedName} en modo{' '}
                                {mode === 'counts' ? 'conteos' : 'normalizado'} y vista{' '}
                                {viewMode === 'compact' ? 'compacta' : 'completa'}
                            </caption>
                            <thead>
                                <tr>
                                    <th className="confusion-matrix-corner" scope="col">
                                        Real ↓ / Pred. →
                                    </th>
                                    {labels.map((label) => (
                                        <th key={label.index} scope="col" title={label.label}>
                                            <span>#{label.index}</span>
                                            <strong>{label.label}</strong>
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {visibleRows.map(({ sourceIndex, values }) => (
                                    <tr key={labels[sourceIndex]?.index ?? sourceIndex}>
                                        <th scope="row" title={labels[sourceIndex]?.label}>
                                            <span>#{labels[sourceIndex]?.index}</span>
                                            <strong>{labels[sourceIndex]?.label}</strong>
                                        </th>
                                        {values.map((value, columnIndex) => {
                                            const realLabel = labels[sourceIndex]?.label
                                            const predictedLabel = labels[columnIndex]?.label
                                            const title = `${realLabel} → ${predictedLabel}: ${formatCellValue(
                                                value,
                                                mode,
                                            )}`

                                            return (
                                                <td
                                                    key={`${labels[sourceIndex]?.index}-${labels[columnIndex]?.index}`}
                                                    className={
                                                        sourceIndex === columnIndex
                                                            ? 'diagonal'
                                                            : ''
                                                    }
                                                    style={getCellStyle(
                                                        value,
                                                        maximum,
                                                        mode,
                                                        variant,
                                                    )}
                                                    title={title}
                                                >
                                                    {formatCellValue(value, mode)}
                                                </td>
                                            )
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div className="confusion-matrix-legend" aria-label="Escala de intensidad">
                    <span>Menor</span>
                    <div className={variant} />
                    <span>Mayor</span>
                </div>
            </section>

            <div className="evaluation-info-banner">
                <strong>Nota:</strong>
                <span>
                    También aparecen clases que solo fueron predichas por el modelo, aunque no
                    existan como etiqueta real en el dataset. La vista compacta conserva esas
                    columnas y oculta únicamente filas sin muestras reales.
                </span>
            </div>
        </div>
    )
}

export default EvaluationConfusionMatrices
