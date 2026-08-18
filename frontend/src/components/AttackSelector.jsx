import {
    ATTACK_CATALOG,
    isAttackSupportedForTask,
} from '../utils/attackCatalog'

function AttackSelector({
    selectedAttack,
    onSelectAttack,
    attackParameters,
    onChangeAttackParameters,
    activeModel,
}) {
    const activeTaskType = activeModel?.task_type || 'classification'
    function handleParameterChange(name, value) {
        onChangeAttackParameters({
            ...attackParameters,
            [name]: Number(value),
        })
    }

    return (
        <div className="sidebar-block">
            <div className="sidebar-block-title">Ataques</div>

            <div className="sidebar-list">
                {ATTACK_CATALOG.map((attack) => {
                    const isSupported = isAttackSupportedForTask(attack.id, activeTaskType)

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
                                if (isSupported) {
                                    onSelectAttack(attack.id)
                                }
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

            {activeTaskType === 'detection' && (
                <p className="muted-text">
                    FGSM y PGD están disponibles para clasificación. Para detección puedes usar Random Noise o YOLO Confidence.
                </p>
            )}

            <div className="attack-config">
                <h4>Configuración</h4>

                <label>
                    Epsilon: {attackParameters.epsilon}
                    <input
                        type="range"
                        min="0"
                        max="0.1"
                        step="0.01"
                        value={attackParameters.epsilon}
                        onChange={(event) =>
                            handleParameterChange('epsilon', event.target.value)
                        }
                    />
                </label>

                {['pgd', 'yolo_confidence'].includes(selectedAttack) && (
                    <>
                        <label>
                            Alpha: {attackParameters.alpha}
                            <input
                                type="range"
                                min="0.001"
                                max="0.05"
                                step="0.001"
                                value={attackParameters.alpha}
                                onChange={(event) =>
                                    handleParameterChange('alpha', event.target.value)
                                }
                            />
                        </label>

                        <label>
                            Steps
                            <input
                                type="number"
                                min="1"
                                max="100"
                                value={attackParameters.steps}
                                onChange={(event) =>
                                    handleParameterChange('steps', event.target.value)
                                }
                            />
                        </label>
                    </>
                )}
            </div>
        </div>
    )
}

export default AttackSelector