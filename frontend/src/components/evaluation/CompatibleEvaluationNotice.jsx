import {
    formatCreatedAgo,
    formatDateTime,
    formatRemainingTime,
    isNearExpiration,
} from '../../utils/evaluationHistory'

function CompatibleEvaluationNotice({
    item,
    busy = false,
    onReuse,
    onRunAgain,
    onRefresh,
    onDismiss,
}) {
    if (!item) return null

    const running = item.status === 'running'
    const nearExpiration = isNearExpiration(item)

    return (
        <section
            className={`evaluation-reuse-notice ${nearExpiration ? 'near-expiration' : ''}`}
            role="status"
        >
            <div className="evaluation-reuse-copy">
                <span className="evaluation-eyebrow">
                    {running ? 'Evaluación en curso' : 'Resultado compatible'}
                </span>
                <h4>
                    {running
                        ? 'Ya se está calculando esta misma configuración'
                        : `Existe una evaluación idéntica de ${formatCreatedAgo(item.created_at)}`}
                </h4>
                {running ? (
                    <p>
                        Para evitar trabajo duplicado no se iniciará otra ejecución
                        simultánea. Comprueba de nuevo cuando haya terminado.
                    </p>
                ) : (
                    <p>
                        Caduca en <strong>{formatRemainingTime(item.expires_at)}</strong>{' '}
                        ({formatDateTime(item.expires_at)}).
                        {nearExpiration && ' Está próxima a caducar; puede convenirte repetirla.'}
                    </p>
                )}
            </div>

            <div className="evaluation-reuse-actions">
                {running ? (
                    <button
                        type="button"
                        className="primary-action"
                        onClick={onRefresh}
                        disabled={busy}
                    >
                        Comprobar de nuevo
                    </button>
                ) : (
                    <>
                        <button
                            type="button"
                            className="primary-action"
                            onClick={onReuse}
                            disabled={busy}
                        >
                            Usar resultado existente
                        </button>
                        <button
                            type="button"
                            className="secondary-action"
                            onClick={onRunAgain}
                            disabled={busy}
                        >
                            Ejecutar de nuevo
                        </button>
                    </>
                )}
                <button
                    type="button"
                    className="text-action"
                    onClick={onDismiss}
                    disabled={busy}
                >
                    Cancelar
                </button>
            </div>
        </section>
    )
}

export default CompatibleEvaluationNotice
