import {
    getAttackDefinition,
    getSupportedAttacksForTask,
    isAttackSupportedForTask,
} from '../utils/attackCatalog'

function parseParameterValue(parameter, event) {
    if (parameter.value_type === 'boolean') {
        return parameter.control === 'checkbox'
            ? event.target.checked
            : event.target.value === 'true'
    }
    if (parameter.value_type === 'integer') return Number.parseInt(event.target.value, 10)
    if (parameter.value_type === 'float') return Number(event.target.value)
    return event.target.value
}

function ParameterControl({ parameter, value, onChange }) {
    if (parameter.control === 'checkbox') {
        return (
            <label title={parameter.description}>
                <input
                    type="checkbox"
                    checked={Boolean(value)}
                    onChange={(event) => onChange(parseParameterValue(parameter, event))}
                />
                {parameter.label}
            </label>
        )
    }

    if (parameter.control === 'select') {
        return (
            <label title={parameter.description}>
                {parameter.label}
                <select
                    value={value}
                    onChange={(event) => onChange(parseParameterValue(parameter, event))}
                >
                    {parameter.choices.map((choice) => (
                        <option key={String(choice.value)} value={choice.value}>
                            {choice.label}
                        </option>
                    ))}
                </select>
            </label>
        )
    }

    const isRange = parameter.control === 'range'
    return (
        <label title={parameter.description}>
            {parameter.label}{isRange ? `: ${value}` : ''}
            <input
                type={isRange ? 'range' : 'number'}
                min={parameter.minimum ?? undefined}
                max={parameter.maximum ?? undefined}
                step={parameter.step ?? undefined}
                value={value}
                onChange={(event) => onChange(parseParameterValue(parameter, event))}
            />
        </label>
    )
}

function AttackSelector({
    selectedAttack,
    onSelectAttack,
    attackParameters,
    onChangeAttackParameters,
    activeModel,
    attackCatalog,
    loading = false,
    error = '',
}) {
    const activeTaskType = activeModel?.task_type || 'classification'
    const selectedDefinition = getAttackDefinition(attackCatalog, selectedAttack)
    const supportedAttacks = getSupportedAttacksForTask(attackCatalog, activeTaskType)

    function handleParameterChange(name, value) {
        onChangeAttackParameters({
            ...attackParameters,
            [name]: value,
        })
    }

    return (
        <div className="sidebar-block">
            <div className="sidebar-block-title">Ataques</div>

            {loading && <p className="muted-text">Cargando catálogo de ataques…</p>}
            {error && <p className="error">No se pudo cargar el catálogo: {error}</p>}

            <div className="sidebar-list">
                {attackCatalog.map((attack) => {
                    const isSupported = isAttackSupportedForTask(
                        attackCatalog,
                        attack.id,
                        activeTaskType,
                    )

                    return (
                        <button
                            key={attack.id}
                            type="button"
                            className={
                                selectedAttack === attack.id
                                    ? 'sidebar-list-item active'
                                    : 'sidebar-list-item'
                            }
                            onClick={() => {
                                if (isSupported) onSelectAttack(attack.id)
                            }}
                            disabled={!isSupported}
                            title={
                                isSupported
                                    ? attack.description
                                    : 'Ataque no disponible para este tipo de modelo'
                            }
                        >
                            <span>{attack.name}</span>

                            {!isSupported && (
                                <small className="attack-unavailable">
                                    No disponible
                                </small>
                            )}
                        </button>
                    )
                })}
            </div>

            {attackCatalog.length > 0 && (
                <p className="muted-text">
                    Compatibles con {activeTaskType}:{' '}
                    {supportedAttacks.map((attack) => attack.name).join(', ') || 'ninguno'}.
                </p>
            )}

            {selectedDefinition && (
                <div className="attack-config">
                    <h4>Configuración</h4>

                    {selectedDefinition.parameters.length === 0 ? (
                        <p className="muted-text">Este ataque no requiere parámetros.</p>
                    ) : (
                        selectedDefinition.parameters.map((parameter) => (
                            <ParameterControl
                                key={parameter.key}
                                parameter={parameter}
                                value={
                                    Object.hasOwn(attackParameters, parameter.key)
                                        ? attackParameters[parameter.key]
                                        : parameter.default
                                }
                                onChange={(value) =>
                                    handleParameterChange(parameter.key, value)
                                }
                            />
                        ))
                    )}
                </div>
            )}
        </div>
    )
}

export default AttackSelector
