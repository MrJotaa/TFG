const ATTACKS = [
    {
        id: 'fgsm',
        name: 'FGSM',
        description: 'Ataque rápido basado en gradiente',
    },
    {
        id: 'random_noise',
        name: 'Random Noise',
        description: 'Baseline con ruido aleatorio',
    },
    {
        id: 'pgd',
        name: 'PGD',
        description: 'Ataque iterativo basado en gradiente',
    },
]

function AttackSelector({
    selectedAttack,
    onSelectAttack,
    attackParameters,
    onChangeAttackParameters,
}) {
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
                {ATTACKS.map((attack) => (
                    <button
                        key={attack.id}
                        type="button"
                        className={
                            selectedAttack === attack.id
                                ? 'sidebar-list-item active'
                                : 'sidebar-list-item'
                        }
                        onClick={() => onSelectAttack(attack.id)}
                    >
                        <span>{attack.name}</span>
                    </button>
                ))}
            </div>

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

                {selectedAttack === 'pgd' && (
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