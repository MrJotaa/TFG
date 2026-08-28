import { useEffect } from 'react'
import {
    formatAttackParameters,
    getAttackName,
} from '../../utils/attackFormatting'
import { individualComparisonKindCopy } from '../../utils/individualAnalysisFlow'

function DialogShell({ children, onClose, busy = false, labelledBy }) {
    useEffect(() => {
        if (busy) return undefined
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') onClose()
        }
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [busy, onClose])

    return (
        <div
            className="individual-analysis-flow-backdrop"
            role="presentation"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget && !busy) onClose()
            }}
        >
            <section
                className="individual-analysis-flow-dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby={labelledBy}
            >
                {children}
            </section>
        </div>
    )
}

export function IndividualAnalysisActionBar({
    historyMode,
    saveState,
    saving,
    onSave,
    onCompare,
}) {
    const saveCopy = saveState === 'saved'
        ? 'Guardado'
        : saveState === 'equivalent'
            ? 'Equivalente ya guardado'
            : saving
                ? 'Guardando…'
                : 'Guardar análisis'

    return (
        <section className="individual-analysis-persistence-bar" aria-label="Acciones del análisis individual">
            <div>
                <span className="evaluation-eyebrow">Resultado individual</span>
                <strong>
                    {historyMode || saveState === 'saved'
                        ? 'Este análisis está conservado en el historial'
                        : saveState === 'equivalent'
                            ? 'Ya existe una ejecución equivalente guardada'
                            : 'Este resultado sigue siendo temporal'}
                </strong>
                <small>
                    {historyMode || saveState === 'saved'
                        ? 'Puedes volver a abrirlo y compararlo sin recalcular el ataque.'
                        : saveState === 'equivalent'
                            ? 'El recálculo actual no se ha fijado: solo el resultado equivalente anterior es permanente.'
                            : 'Guárdalo únicamente si este caso merece conservarse indefinidamente.'}
                </small>
            </div>
            <div className="individual-analysis-persistence-actions">
                {!historyMode && (
                    <button
                        type="button"
                        className="secondary-action"
                        onClick={onSave}
                        disabled={saving || saveState === 'saved' || saveState === 'equivalent'}
                    >
                        {saveCopy}
                    </button>
                )}
                <button
                    type="button"
                    className="primary-action"
                    onClick={onCompare}
                    disabled={saving}
                >
                    Comparar este análisis
                </button>
            </div>
        </section>
    )
}

export function DuplicateAnalysisDialog({
    match,
    attackCatalog,
    preparingB,
    busy,
    onOpenSaved,
    onUseAsB,
    onRecalculate,
    onCancel,
}) {
    if (!match) return null

    return (
        <DialogShell
            onClose={onCancel}
            busy={busy}
            labelledBy="duplicate-analysis-title"
        >
            <span className="evaluation-eyebrow">Cálculo ya disponible</span>
            <h2 id="duplicate-analysis-title">Este análisis ya está en tu historial</h2>
            <p>
                Coinciden imagen, revisión del dataset, modelo, ataque, parámetros y
                versiones internas. No necesitas repetir un ataque potencialmente costoso
                salvo que quieras recalcularlo de forma explícita.
            </p>
            <div className="individual-analysis-duplicate-summary">
                <strong>{match.model_name}</strong>
                <span>{getAttackName(match.attack_type, attackCatalog)}</span>
                <small>
                    {formatAttackParameters(match.attack_type, match.parameters, attackCatalog)}
                </small>
            </div>
            <div className="individual-analysis-flow-actions wrap">
                <button type="button" className="secondary-action" onClick={onCancel} disabled={busy}>
                    Cancelar
                </button>
                <button type="button" className="secondary-action" onClick={onOpenSaved} disabled={busy}>
                    Abrir guardado
                </button>
                {preparingB && (
                    <button type="button" className="primary-action" onClick={onUseAsB} disabled={busy}>
                        Usar guardado como B
                    </button>
                )}
                <button type="button" className="danger-outline-action" onClick={onRecalculate} disabled={busy}>
                    Calcular de nuevo
                </button>
            </div>
        </DialogShell>
    )
}

export function IndividualAnalysisCompareDialog({
    baseItem,
    candidates,
    loadingCandidates,
    attackCatalog,
    onPrepare,
    onCompareCandidate,
    onClose,
}) {
    return (
        <DialogShell onClose={onClose} labelledBy="individual-analysis-compare-title">
            <span className="evaluation-eyebrow">Comparación individual</span>
            <h2 id="individual-analysis-compare-title">¿Qué quieres cambiar en B?</h2>
            <p>
                La imagen y la revisión del dataset permanecen fijas. Para que la comparación
                sea controlada solo puede cambiar una variable experimental.
            </p>

            <div className="individual-analysis-compare-base">
                <span>A</span>
                <div>
                    <strong>{baseItem.model_name}</strong>
                    <small>
                        {getAttackName(baseItem.attack_type, attackCatalog)} ·{' '}
                        {formatAttackParameters(baseItem.attack_type, baseItem.parameters, attackCatalog)}
                    </small>
                </div>
            </div>

            <div className="individual-analysis-compare-options">
                {['attack', 'attack_configuration', 'model'].map((kind) => {
                    const copy = individualComparisonKindCopy(kind)
                    return (
                        <button key={kind} type="button" onClick={() => onPrepare(kind)}>
                            <strong>{copy.label}</strong>
                            <span>{copy.instruction}</span>
                        </button>
                    )
                })}
            </div>

            <section className="individual-analysis-saved-candidates">
                <header>
                    <div>
                        <strong>Usar un análisis compatible ya guardado</strong>
                        <span>Evita cualquier recálculo si B ya existe en el historial.</span>
                    </div>
                    {loadingCandidates && <small>Buscando…</small>}
                </header>
                {!loadingCandidates && candidates.length === 0 ? (
                    <p>No hay otro análisis guardado compatible para esta imagen.</p>
                ) : (
                    <div>
                        {candidates.slice(0, 6).map(({ item, comparison }) => (
                            <button
                                key={item.analysis_id}
                                type="button"
                                onClick={() => onCompareCandidate(item)}
                            >
                                <span className="individual-comparison-side-token">B</span>
                                <div>
                                    <strong>{item.model_name}</strong>
                                    <small>
                                        {getAttackName(item.attack_type, attackCatalog)} · {comparison.label}
                                    </small>
                                </div>
                                <span>Comparar →</span>
                            </button>
                        ))}
                    </div>
                )}
            </section>

            <div className="individual-analysis-flow-actions">
                <button type="button" className="secondary-action" onClick={onClose}>
                    Cancelar
                </button>
            </div>
        </DialogShell>
    )
}
