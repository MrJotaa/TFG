import {
    getAttackDefinition,
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
            <label className="parameter-toggle" title={parameter.description}>
                <span className="parameter-toggle-copy">
                    <strong>{parameter.label}</strong>
                    {parameter.description && <small>{parameter.description}</small>}
                </span>
                <span className="parameter-switch-control">
                    <input
                        type="checkbox"
                        checked={Boolean(value)}
                        onChange={(event) => onChange(parseParameterValue(parameter, event))}
                    />
                    <span className="parameter-switch" aria-hidden="true" />
                </span>
            </label>
        )
    }

    if (parameter.control === 'select') {
        return (
            <label className="parameter-field" title={parameter.description}>
                <span>{parameter.label}</span>
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

    if (isRange) {
        return (
            <label className="parameter-field parameter-range" title={parameter.description}>
                <span className="parameter-field-heading">
                    <span>{parameter.label}</span>
                    <output>{String(value)}</output>
                </span>
                <input
                    type="range"
                    min={parameter.minimum ?? undefined}
                    max={parameter.maximum ?? undefined}
                    step={parameter.step ?? undefined}
                    value={value}
                    onChange={(event) => onChange(parseParameterValue(parameter, event))}
                />
            </label>
        )
    }

    return (
        <label className="parameter-field" title={parameter.description}>
            <span>{parameter.label}</span>
            <input
                type="number"
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

    function handleParameterChange(name, value) {
        onChangeAttackParameters({
            ...attackParameters,
            [name]: value,
        })
    }

    return (
        <section className="sidebar-block dashboard-config-section attack-selector-section">
            <div className="dashboard-config-heading">
                <div>
                    <span className="dashboard-section-eyebrow">Configuración</span>
                    <h2>Ataque</h2>
                </div>
                <span className="dashboard-config-step">02</span>
            </div>

            {loading && <p className="muted-text">Cargando catálogo de ataques…</p>}
            {error && <p className="error">No se pudo cargar el catálogo: {error}</p>}

            <div className="sidebar-list attack-list" aria-label="Ataques disponibles">
                {attackCatalog.map((attack) => {
                    const isSupported = isAttackSupportedForTask(
                        attackCatalog,
                        attack.id,
                        activeTaskType,
                    )
                    const isSelected = selectedAttack === attack.id

                    return (
                        <button
                            key={attack.id}
                            type="button"
                            className={isSelected ? 'sidebar-list-item active' : 'sidebar-list-item'}
                            onClick={() => {
                                if (isSupported) onSelectAttack(attack.id)
                            }}
                            disabled={!isSupported}
                            title={
                                isSupported
                                    ? attack.description
                                    : `No compatible con ${activeTaskType === 'classification' ? 'clasificación' : 'detección'}`
                            }
                            aria-pressed={isSupported ? isSelected : undefined}
                        >
                            <span>{attack.name}</span>

                            {!isSupported && (
                                <small className="attack-unavailable">No disponible</small>
                            )}
                        </button>
                    )
                })}
            </div>

            {selectedDefinition && (
                <div className="attack-config dashboard-parameter-section">
                    <div className="dashboard-parameter-heading">
                        <span className="dashboard-section-eyebrow">Parámetros</span>
                        <strong>{selectedDefinition.name}</strong>
                    </div>

                    {selectedDefinition.parameters.length === 0 ? (
                        <p className="muted-text">Este ataque no requiere parámetros.</p>
                    ) : (
                        <div className="parameter-control-list">
                            {selectedDefinition.parameters.map((parameter) => (
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
                            ))}
                        </div>
                    )}
                </div>
            )}
        </section>
    )
}

export default AttackSelector
