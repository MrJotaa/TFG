import { useMemo, useRef, useState } from 'react'
import { resolveApiUrl } from '../../api/client'
import {
    clampExplorerZoom,
    EXPLORER_MAX_ZOOM,
    EXPLORER_MIN_ZOOM,
    EXPLORER_ZOOM_STEP,
    resolveOverlaySplit,
} from '../../utils/adversarialExplorer'
import {
    comparisonExplanationState,
    getComparisonExplanationMessage,
    getComparisonKeyboardAction,
    getComparisonLayerCaption,
    comparisonLayerOptions,
    resolveIndividualAnalysisComparisonSource,
} from '../../utils/individualAnalysisComparison'


function ComparisonViewport({
    label,
    source,
    zoom,
    pan,
    handlers,
    side,
}) {
    return (
        <figure className={`individual-comparison-figure side-${side.toLowerCase()}`}>
            <figcaption>
                <span className="individual-comparison-side-token">{side}</span>
                <strong>{label}</strong>
            </figcaption>
            <button
                type="button"
                className={`individual-comparison-viewport ${zoom > 1 ? 'is-zoomed' : ''}`}
                aria-label={`${label}. Usa las flechas para desplazar y más o menos para ampliar.`}
                {...handlers}
            >
                {source ? (
                    <img
                        src={resolveApiUrl(source)}
                        alt={label}
                        draggable="false"
                        style={{
                            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                        }}
                    />
                ) : (
                    <span className="individual-comparison-missing">Artefacto no disponible</span>
                )}
            </button>
        </figure>
    )
}

function ComparisonOverlay({
    leftSource,
    rightSource,
    zoom,
    pan,
    split,
    handlers,
    onSplitChange,
}) {
    const splitState = resolveOverlaySplit(split)

    return (
        <div className="individual-comparison-overlay">
            <button
                type="button"
                className={`individual-comparison-viewport overlay ${zoom > 1 ? 'is-zoomed' : ''}`}
                aria-label="Superposición A y B. Usa las flechas para desplazar y más o menos para ampliar."
                {...handlers}
            >
                {rightSource && (
                    <img
                        src={resolveApiUrl(rightSource)}
                        alt="Evaluación B"
                        draggable="false"
                        style={{
                            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                        }}
                    />
                )}
                {leftSource && (
                    <span
                        className="individual-comparison-overlay-layer"
                        style={{ clipPath: splitState.originalClipPath }}
                    >
                        <img
                            src={resolveApiUrl(leftSource)}
                            alt="Evaluación A"
                            draggable="false"
                            style={{
                                transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                            }}
                        />
                    </span>
                )}
                <span
                    className="individual-comparison-overlay-divider"
                    style={{ left: `${splitState.originalPercent}%` }}
                    aria-hidden="true"
                />
                <span className="individual-comparison-overlay-label side-a">A</span>
                <span className="individual-comparison-overlay-label side-b">B</span>
            </button>
            <label className="individual-comparison-split-control">
                <span>A {splitState.originalPercent}%</span>
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={split}
                    onChange={onSplitChange}
                    aria-label="Posición del divisor entre los análisis A y B"
                />
                <output>{splitState.adversarialPercent}% B</output>
            </label>
        </div>
    )
}

function ComparisonToolbar({
    layout,
    setLayout,
    layer,
    setLayer,
    layerOptions,
    stage,
    setStage,
    taskType,
    showAnnotations,
    setShowAnnotations,
    zoom,
    changeZoom,
    resetView,
}) {
    return (
        <div className="individual-comparison-toolbar">
            <div className="individual-comparison-control-group">
                <span>Disposición</span>
                <fieldset className="individual-comparison-segmented semantic-fieldset-reset" aria-label="Disposición de la comparación">
                    <button type="button" className={layout === 'parallel' ? 'active' : ''} onClick={() => setLayout('parallel')}>Paralelo</button>
                    <button type="button" className={layout === 'overlay' ? 'active' : ''} onClick={() => setLayout('overlay')}>Superpuesto</button>
                </fieldset>
            </div>

            <div className="individual-comparison-control-group">
                <span>Capa</span>
                <fieldset className="individual-comparison-segmented semantic-fieldset-reset" aria-label="Capa visual">
                    {layerOptions.map((option) => (
                        <button key={option.id} type="button" className={layer === option.id ? 'active' : ''} onClick={() => setLayer(option.id)}>{option.label}</button>
                    ))}
                </fieldset>
            </div>

            {layer !== 'perturbation' && (
                <div className="individual-comparison-control-group">
                    <span>Estado</span>
                    <fieldset className="individual-comparison-segmented semantic-fieldset-reset" aria-label="Estado de la imagen">
                        <button type="button" className={stage === 'original' ? 'active' : ''} onClick={() => setStage('original')}>Original</button>
                        <button type="button" className={stage === 'adversarial' ? 'active' : ''} onClick={() => setStage('adversarial')}>Tras ataque</button>
                    </fieldset>
                </div>
            )}

            {taskType === 'detection' && layer === 'image' && (
                <label className="individual-comparison-toggle">
                    <input type="checkbox" checked={showAnnotations} onChange={(event) => setShowAnnotations(event.target.checked)} />
                    <span>Mostrar cajas</span>
                </label>
            )}

            <fieldset className="individual-comparison-zoom semantic-fieldset-reset" aria-label="Zoom sincronizado">
                <button type="button" disabled={zoom === EXPLORER_MIN_ZOOM} onClick={() => changeZoom(zoom - EXPLORER_ZOOM_STEP)} aria-label="Reducir">−</button>
                <output>{Math.round(zoom * 100)}%</output>
                <button type="button" disabled={zoom === EXPLORER_MAX_ZOOM} onClick={() => changeZoom(zoom + EXPLORER_ZOOM_STEP)} aria-label="Ampliar">+</button>
                <button type="button" onClick={resetView}>Restablecer</button>
            </fieldset>
        </div>
    )
}

function ComparisonCanvas({ layout, leftLabel, rightLabel, leftSource, rightSource, zoom, pan, split, handlers, onSplitChange }) {
    if (layout === 'overlay') {
        return <ComparisonOverlay leftSource={leftSource} rightSource={rightSource} zoom={zoom} pan={pan} split={split} handlers={handlers} onSplitChange={onSplitChange} />
    }
    return (
        <div className="individual-comparison-grid">
            <ComparisonViewport side="A" label={leftLabel} source={leftSource} zoom={zoom} pan={pan} handlers={handlers} />
            <ComparisonViewport side="B" label={rightLabel} source={rightSource} zoom={zoom} pan={pan} handlers={handlers} />
        </div>
    )
}

function ComparisonExplanationNote({ comparable, taskType, comparisonKind, explanationState }) {
    if (comparable) return null
    return (
        <p className="individual-comparison-explanation-note">
            <strong>Explicabilidad:</strong>{' '}
            {getComparisonExplanationMessage({ taskType, comparisonKind, explanationState })}
        </p>
    )
}

function IndividualAnalysisComparisonExplorer({
    taskType,
    image,
    leftAnalysis,
    rightAnalysis,
    leftLabel,
    rightLabel,
    comparisonKind,
}) {
    const [layout, setLayout] = useState('parallel')
    const [layer, setLayer] = useState('image')
    const [stage, setStage] = useState('adversarial')
    const [showAnnotations, setShowAnnotations] = useState(taskType === 'detection')
    const [zoom, setZoom] = useState(1)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [split, setSplit] = useState(50)
    const dragRef = useRef(null)

    const layerOptions = useMemo(
        () => comparisonLayerOptions(taskType, leftAnalysis, rightAnalysis),
        [taskType, leftAnalysis, rightAnalysis],
    )
    const explanationState = useMemo(
        () => comparisonExplanationState(taskType, leftAnalysis, rightAnalysis),
        [taskType, leftAnalysis, rightAnalysis],
    )

    const effectiveStage = layer === 'perturbation' ? 'adversarial' : stage
    const leftSource = resolveIndividualAnalysisComparisonSource({
        taskType,
        analysis: leftAnalysis,
        image,
        layer,
        stage: effectiveStage,
        showAnnotations,
    })
    const rightSource = resolveIndividualAnalysisComparisonSource({
        taskType,
        analysis: rightAnalysis,
        image,
        layer,
        stage: effectiveStage,
        showAnnotations,
    })

    function resetView() {
        setZoom(1)
        setPan({ x: 0, y: 0 })
    }

    function changeZoom(value) {
        const next = clampExplorerZoom(value)
        setZoom(next)
        if (next === EXPLORER_MIN_ZOOM) setPan({ x: 0, y: 0 })
    }

    function handlePointerDown(event) {
        if (zoom === EXPLORER_MIN_ZOOM || event.button !== 0) return
        event.currentTarget.setPointerCapture(event.pointerId)
        dragRef.current = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            panX: pan.x,
            panY: pan.y,
        }
    }

    function handlePointerMove(event) {
        const drag = dragRef.current
        if (!drag || drag.pointerId !== event.pointerId) return
        setPan({
            x: drag.panX + event.clientX - drag.startX,
            y: drag.panY + event.clientY - drag.startY,
        })
    }

    function handlePointerUp(event) {
        if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null
    }

    function handleKeyDown(event) {
        const action = getComparisonKeyboardAction(event.key, zoom, EXPLORER_MIN_ZOOM)
        if (!action) return
        event.preventDefault()
        if (action.type === 'pan') {
            setPan((current) => ({ x: current.x + action.x, y: current.y + action.y }))
            return
        }
        if (action.type === 'zoom') {
            changeZoom(zoom + action.direction * EXPLORER_ZOOM_STEP)
            return
        }
        resetView()
    }

    const handlers = {
        onKeyDown: handleKeyDown,
        onPointerDown: handlePointerDown,
        onPointerMove: handlePointerMove,
        onPointerUp: handlePointerUp,
        onPointerCancel: handlePointerUp,
    }

    const layerCaption = getComparisonLayerCaption(layer, taskType, stage)

    return (
        <section className="individual-comparison-explorer" aria-labelledby="individual-comparison-explorer-title">
            <header className="individual-comparison-explorer-heading">
                <div>
                    <span className="evaluation-eyebrow">Exploración visual sincronizada</span>
                    <h2 id="individual-comparison-explorer-title">La misma imagen bajo A y B</h2>
                    <p>
                        Cambia de capa sin perder el encuadre. Zoom, desplazamiento y
                        superposición se mantienen sincronizados entre los dos resultados.
                    </p>
                </div>
                <div className="individual-comparison-image-identity">
                    <strong>{image?.filename || 'Imagen analizada'}</strong>
                    <span>{layerCaption}</span>
                </div>
            </header>

            <ComparisonToolbar
                layout={layout}
                setLayout={setLayout}
                layer={layer}
                setLayer={setLayer}
                layerOptions={layerOptions}
                stage={stage}
                setStage={setStage}
                taskType={taskType}
                showAnnotations={showAnnotations}
                setShowAnnotations={setShowAnnotations}
                zoom={zoom}
                changeZoom={changeZoom}
                resetView={resetView}
            />

            <ComparisonCanvas
                layout={layout}
                leftLabel={leftLabel}
                rightLabel={rightLabel}
                leftSource={leftSource}
                rightSource={rightSource}
                zoom={zoom}
                pan={pan}
                split={split}
                handlers={handlers}
                onSplitChange={(event) => setSplit(Number(event.target.value))}
            />

            <div className="individual-comparison-explorer-footer">
                <span>Con zoom activo puedes arrastrar cualquiera de las vistas.</span>
                <span>Teclado: flechas · + / − · 0 para restablecer</span>
            </div>

            <ComparisonExplanationNote
                comparable={explanationState.comparable}
                taskType={taskType}
                comparisonKind={comparisonKind}
                explanationState={explanationState}
            />
        </section>
    )
}

export default IndividualAnalysisComparisonExplorer
