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
    comparisonLayerOptions,
    resolveIndividualAnalysisComparisonSource,
} from '../../utils/individualAnalysisComparison'

const PAN_STEP = 24

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
        const movements = {
            ArrowLeft: { x: PAN_STEP, y: 0 },
            ArrowRight: { x: -PAN_STEP, y: 0 },
            ArrowUp: { x: 0, y: PAN_STEP },
            ArrowDown: { x: 0, y: -PAN_STEP },
        }
        if (movements[event.key] && zoom > EXPLORER_MIN_ZOOM) {
            event.preventDefault()
            setPan((current) => ({
                x: current.x + movements[event.key].x,
                y: current.y + movements[event.key].y,
            }))
            return
        }
        if (event.key === '+' || event.key === '=') {
            event.preventDefault()
            changeZoom(zoom + EXPLORER_ZOOM_STEP)
        } else if (event.key === '-') {
            event.preventDefault()
            changeZoom(zoom - EXPLORER_ZOOM_STEP)
        } else if (event.key === '0') {
            event.preventDefault()
            resetView()
        }
    }

    const handlers = {
        onKeyDown: handleKeyDown,
        onPointerDown: handlePointerDown,
        onPointerMove: handlePointerMove,
        onPointerUp: handlePointerUp,
        onPointerCancel: handlePointerUp,
    }

    const layerCaption = layer === 'perturbation'
        ? 'Mapa amplificado de la perturbación'
        : layer === 'explanation'
            ? taskType === 'detection' ? 'Eigen-CAM persistido' : 'Grad-CAM persistido'
            : stage === 'original' ? 'Resultado original' : 'Resultado tras el ataque'

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

            <div className="individual-comparison-toolbar">
                <div className="individual-comparison-control-group">
                    <span>Disposición</span>
                    <div className="individual-comparison-segmented" role="group" aria-label="Disposición de la comparación">
                        <button
                            type="button"
                            className={layout === 'parallel' ? 'active' : ''}
                            onClick={() => setLayout('parallel')}
                        >
                            Paralelo
                        </button>
                        <button
                            type="button"
                            className={layout === 'overlay' ? 'active' : ''}
                            onClick={() => setLayout('overlay')}
                        >
                            Superpuesto
                        </button>
                    </div>
                </div>

                <div className="individual-comparison-control-group">
                    <span>Capa</span>
                    <div className="individual-comparison-segmented" role="group" aria-label="Capa visual">
                        {layerOptions.map((option) => (
                            <button
                                key={option.id}
                                type="button"
                                className={layer === option.id ? 'active' : ''}
                                onClick={() => setLayer(option.id)}
                            >
                                {option.label}
                            </button>
                        ))}
                    </div>
                </div>

                {layer !== 'perturbation' && (
                    <div className="individual-comparison-control-group">
                        <span>Estado</span>
                        <div className="individual-comparison-segmented" role="group" aria-label="Estado de la imagen">
                            <button
                                type="button"
                                className={stage === 'original' ? 'active' : ''}
                                onClick={() => setStage('original')}
                            >
                                Original
                            </button>
                            <button
                                type="button"
                                className={stage === 'adversarial' ? 'active' : ''}
                                onClick={() => setStage('adversarial')}
                            >
                                Tras ataque
                            </button>
                        </div>
                    </div>
                )}

                {taskType === 'detection' && layer === 'image' && (
                    <label className="individual-comparison-toggle">
                        <input
                            type="checkbox"
                            checked={showAnnotations}
                            onChange={(event) => setShowAnnotations(event.target.checked)}
                        />
                        <span>Mostrar cajas</span>
                    </label>
                )}

                <div className="individual-comparison-zoom" role="group" aria-label="Zoom sincronizado">
                    <button
                        type="button"
                        disabled={zoom === EXPLORER_MIN_ZOOM}
                        onClick={() => changeZoom(zoom - EXPLORER_ZOOM_STEP)}
                        aria-label="Reducir"
                    >
                        −
                    </button>
                    <output>{Math.round(zoom * 100)}%</output>
                    <button
                        type="button"
                        disabled={zoom === EXPLORER_MAX_ZOOM}
                        onClick={() => changeZoom(zoom + EXPLORER_ZOOM_STEP)}
                        aria-label="Ampliar"
                    >
                        +
                    </button>
                    <button type="button" onClick={resetView}>Restablecer</button>
                </div>
            </div>

            {layout === 'parallel' ? (
                <div className="individual-comparison-grid">
                    <ComparisonViewport
                        side="A"
                        label={leftLabel}
                        source={leftSource}
                        zoom={zoom}
                        pan={pan}
                        handlers={handlers}
                    />
                    <ComparisonViewport
                        side="B"
                        label={rightLabel}
                        source={rightSource}
                        zoom={zoom}
                        pan={pan}
                        handlers={handlers}
                    />
                </div>
            ) : (
                <ComparisonOverlay
                    leftSource={leftSource}
                    rightSource={rightSource}
                    zoom={zoom}
                    pan={pan}
                    split={split}
                    handlers={handlers}
                    onSplitChange={(event) => setSplit(Number(event.target.value))}
                />
            )}

            <div className="individual-comparison-explorer-footer">
                <span>Con zoom activo puedes arrastrar cualquiera de las vistas.</span>
                <span>Teclado: flechas · + / − · 0 para restablecer</span>
            </div>

            {!explanationState.comparable && (
                <p className="individual-comparison-explanation-note">
                    <strong>Explicabilidad:</strong>{' '}
                    {comparisonKind === 'model'
                        ? `la capa solo aparece si ambos análisis conservaron su ${taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'} al ejecutarse. Si falta alguno no se regenera, porque la explicación depende del modelo activo y el historial no cambia ese estado global.`
                        : explanationState.left || explanationState.right
                            ? `solo uno de los análisis conserva ${taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'}; la capa se oculta para evitar una comparación incompleta.`
                            : `ninguno de los dos análisis conserva ${taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'}; el historial no recalcula explicaciones al abrir una comparación.`}
                </p>
            )}
        </section>
    )
}

export default IndividualAnalysisComparisonExplorer
