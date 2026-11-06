import { useEffect, useRef, useState } from 'react'
import {
    getAttackDefinition,
    getSupportedAttacksForTask,
} from '../utils/attackCatalog'
import { resolveScrollOverflow } from '../utils/dashboard'

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
        const controlId = `attack-parameter-${parameter.key}`

        return (
            <div className="parameter-field parameter-range" title={parameter.description}>
                <span className="parameter-field-heading">
                    <label htmlFor={controlId}>{parameter.label}</label>
                    <output htmlFor={controlId}>{String(value)}</output>
                </span>
                <input
                    id={controlId}
                    type="range"
                    min={parameter.minimum ?? undefined}
                    max={parameter.maximum ?? undefined}
                    step={parameter.step ?? undefined}
                    value={value}
                    onChange={(event) => onChange(parseParameterValue(parameter, event))}
                />
            </div>
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
    const availableAttacks = getSupportedAttacksForTask(
        attackCatalog,
        activeTaskType,
        activeModel?.id,
    )
    const selectedDefinition = availableAttacks.some((attack) => attack.id === selectedAttack)
        ? getAttackDefinition(attackCatalog, selectedAttack)
        : null
    const attackListRef = useRef(null)
    const [scrollOverflow, setScrollOverflow] = useState({
        canScrollUp: false,
        canScrollDown: false,
    })

    useEffect(() => {
        const element = attackListRef.current
        if (!element) return undefined

        function updateOverflow() {
            setScrollOverflow(resolveScrollOverflow(element))
        }

        const frame = window.requestAnimationFrame(updateOverflow)
        const observer = typeof ResizeObserver === 'function'
            ? new ResizeObserver(updateOverflow)
            : null
        observer?.observe(element)

        return () => {
            window.cancelAnimationFrame(frame)
            observer?.disconnect()
        }
    }, [availableAttacks.length, activeTaskType, activeModel?.id])

    useEffect(() => {
        if (attackListRef.current) attackListRef.current.scrollTop = 0
    }, [activeModel?.id, activeTaskType])

    function handleAttackListScroll(event) {
        setScrollOverflow(resolveScrollOverflow(event.currentTarget))
    }

    function scrollAttackList(direction) {
        attackListRef.current?.scrollBy({
            top: direction * 120,
            behavior: 'smooth',
        })
    }

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

            <div className={`attack-list-shell ${scrollOverflow.canScrollUp ? 'has-up' : ''} ${scrollOverflow.canScrollDown ? 'has-down' : ''}`}>
                {scrollOverflow.canScrollUp && (
                    <button
                        type="button"
                        className="attack-scroll-hint up"
                        onClick={() => scrollAttackList(-1)}
                        aria-label="Ver ataques anteriores"
                        title="Hay más ataques arriba"
                    >
                        <span aria-hidden="true">⌃</span>
                    </button>
                )}

                <div
                    ref={attackListRef}
                    className="sidebar-list attack-list"
                    aria-label="Ataques disponibles"
                    onScroll={handleAttackListScroll}
                >
                    {availableAttacks.map((attack) => (
                        <button
                            key={attack.id}
                            type="button"
                            className={selectedAttack === attack.id
                                ? 'sidebar-list-item active'
                                : 'sidebar-list-item'}
                            onClick={() => onSelectAttack(attack.id)}
                            title={attack.description}
                            aria-pressed={selectedAttack === attack.id}
                        >
                            <span>{attack.name}</span>
                        </button>
                    ))}
                    {!loading && !error && availableAttacks.length === 0 && (
                        <p className="muted-text">No hay ataques compatibles con este modelo.</p>
                    )}
                </div>

                {scrollOverflow.canScrollDown && (
                    <button
                        type="button"
                        className="attack-scroll-hint down"
                        onClick={() => scrollAttackList(1)}
                        aria-label="Ver más ataques"
                        title="Hay más ataques abajo"
                    >
                        <span aria-hidden="true">⌄</span>
                    </button>
                )}
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
