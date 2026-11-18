import { useRef } from 'react'

function EvaluationTabs({ tabs, activeTab, onChange }) {
    const tabRefs = useRef([])

    function selectTab(index) {
        const tab = tabs[index]

        if (!tab) {
            return
        }

        onChange(tab.id)
        tabRefs.current[index]?.focus()
    }

    function handleKeyDown(event, index) {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
            return
        }

        event.preventDefault()

        if (event.key === 'Home') {
            selectTab(0)
            return
        }

        if (event.key === 'End') {
            selectTab(tabs.length - 1)
            return
        }

        const direction = event.key === 'ArrowRight' ? 1 : -1
        const nextIndex = (index + direction + tabs.length) % tabs.length
        selectTab(nextIndex)
    }

    return (
        <div className="evaluation-tabs" role="tablist" aria-label="Secciones de evaluación">
            {tabs.map((tab, index) => {
                const isActive = activeTab === tab.id

                return (
                    <button
                        key={tab.id}
                        ref={(element) => {
                            tabRefs.current[index] = element
                        }}
                        id={`evaluation-tab-${tab.id}`}
                        type="button"
                        role="tab"
                        aria-controls={`evaluation-panel-${tab.id}`}
                        aria-selected={isActive}
                        tabIndex={isActive ? 0 : -1}
                        className={isActive ? 'evaluation-tab active' : 'evaluation-tab'}
                        onClick={() => onChange(tab.id)}
                        onKeyDown={(event) => handleKeyDown(event, index)}
                    >
                        <span>{tab.label}</span>
                        {tab.badge !== undefined && tab.badge !== null && (
                            <small aria-label={`${tab.badge} elementos`}>{tab.badge}</small>
                        )}
                    </button>
                )
            })}
        </div>
    )
}

export default EvaluationTabs
