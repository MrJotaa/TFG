import { useMemo, useRef, useState } from 'react'
import { resolveApiUrl } from '../../api/client'
import {
    buildExplorerSources,
    clampExplorerZoom,
    EXPLORER_MAX_ZOOM,
    EXPLORER_MIN_ZOOM,
    EXPLORER_ZOOM_STEP,
    resolveExplorerPair,
    resolveOverlaySplit,
} from '../../utils/adversarialExplorer'

const PAN_STEP = 24

function ExplorerImage({
    alt,
    imageUrl,
    onKeyDown,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    pan,
    title,
    zoom,
}) {
    return (
        <figure className="adversarial-explorer-figure">
            <figcaption>{title}</figcaption>
            <button
                type="button"
                className={`adversarial-explorer-viewport ${zoom > 1 ? 'is-zoomed' : ''}`}
                aria-label={`${title}. Usa las flechas para desplazar y las teclas más o menos para ampliar.`}
                onDoubleClick={() => onKeyDown({ key: '0', preventDefault() {} })}
                onKeyDown={onKeyDown}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
            >
                {imageUrl ? (
                    <img
                        src={resolveApiUrl(imageUrl)}
                        alt={alt}
                        draggable="false"
                        style={{
                            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                        }}
                    />
                ) : (
                    <span className="adversarial-explorer-missing">
                        Imagen no disponible
                    </span>
                )}
            </button>
        </figure>
    )
}

function OverlayComparison({
    adversarialUrl,
    originalUrl,
    pan,
    split,
    viewportHandlers,
    zoom,
}) {
    const { onSplitChange, ...imageHandlers } = viewportHandlers
    const splitState = resolveOverlaySplit(split)

    return (
        <div className="adversarial-overlay-comparison">
            <button
                type="button"
                className={`adversarial-explorer-viewport overlay ${zoom > 1 ? 'is-zoomed' : ''}`}
                aria-label="Comparación superpuesta. Usa las flechas para desplazar y las teclas más o menos para ampliar."
                {...imageHandlers}
            >
                <img
                    src={resolveApiUrl(adversarialUrl)}
                    alt="Vista adversaria"
                    draggable="false"
                    style={{
                        transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                    }}
                />
                <span
                    className="adversarial-overlay-layer"
                    style={{ clipPath: splitState.originalClipPath }}
                >
                    <img
                        src={resolveApiUrl(originalUrl)}
                        alt="Vista original"
                        draggable="false"
                        style={{
                            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
                        }}
                    />
                </span>
                <span
                    className="adversarial-overlay-divider"
                    style={{ left: `${splitState.originalPercent}%` }}
                    aria-hidden="true"
                />
                <span className="adversarial-overlay-label original">Original</span>
                <span className="adversarial-overlay-label adversarial">
                    Adversaria
                </span>
            </button>

            <label className="adversarial-split-control">
                <span>Original {splitState.originalPercent}%</span>
                <output>{splitState.adversarialPercent}% adversaria</output>
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={split}
                    onChange={onSplitChange}
                    aria-label="Posición del divisor entre la imagen original y la adversaria"
                />
            </label>
        </div>
    )
}

function AdversarialImageExplorer({
    attackResult,
    filename,
    imageDimensions,
    perturbationMetrics = null,
    selectedImage,
    taskType,
}) {
    const [layout, setLayout] = useState('parallel')
    const [layer, setLayer] = useState('image')
    const [showAnnotations, setShowAnnotations] = useState(
        taskType === 'detection',
    )
    const [showPerturbation, setShowPerturbation] = useState(true)
    const [zoom, setZoom] = useState(1)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [split, setSplit] = useState(50)
    const dragRef = useRef(null)

    const sources = useMemo(
        () => buildExplorerSources({ taskType, selectedImage, attackResult }),
        [attackResult, selectedImage, taskType],
    )
    const pair = resolveExplorerPair({ sources, layer, showAnnotations })

    function resetView() {
        setZoom(1)
        setPan({ x: 0, y: 0 })
    }

    function changeZoom(nextZoom) {
        const normalizedZoom = clampExplorerZoom(nextZoom)
        setZoom(normalizedZoom)

        if (normalizedZoom === EXPLORER_MIN_ZOOM) {
            setPan({ x: 0, y: 0 })
        }
    }

    function handlePointerDown(event) {
        if (zoom === EXPLORER_MIN_ZOOM || event.button !== 0) {
            return
        }

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

        if (!drag || drag.pointerId !== event.pointerId) {
            return
        }

        setPan({
            x: drag.panX + event.clientX - drag.startX,
            y: drag.panY + event.clientY - drag.startY,
        })
    }

    function handlePointerUp(event) {
        if (dragRef.current?.pointerId === event.pointerId) {
            dragRef.current = null
        }
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

    const viewportHandlers = {
        onKeyDown: handleKeyDown,
        onPointerDown: handlePointerDown,
        onPointerMove: handlePointerMove,
        onPointerUp: handlePointerUp,
        onPointerCancel: handlePointerUp,
    }

    if (!sources) {
        return null
    }

    return (
        <section className="adversarial-explorer" aria-labelledby="explorer-title">
            <header className="adversarial-explorer-header">
                <div>
                    <span className="evaluation-eyebrow">Exploración visual</span>
                    <h3 id="explorer-title">Original frente al resultado tras el ataque</h3>
                    <p>
                        Inspecciona el mismo encuadre con transformación sincronizada.
                        La imagen se conserva en su geometría de origen.
                    </p>
                </div>
                <div className="adversarial-explorer-identity">
                    <strong>{filename || 'Imagen seleccionada'}</strong>
                    {imageDimensions && <span>{imageDimensions}</span>}
                </div>
            </header>

            <div className="adversarial-explorer-toolbar" aria-label="Controles del explorador">
                <fieldset className="adversarial-explorer-segmented semantic-fieldset-reset" aria-label="Disposición">
                    <button
                        type="button"
                        className={layout === 'parallel' ? 'active' : ''}
                        aria-pressed={layout === 'parallel'}
                        onClick={() => setLayout('parallel')}
                    >
                        Paralelo
                    </button>
                    <button
                        type="button"
                        className={layout === 'overlay' ? 'active' : ''}
                        aria-pressed={layout === 'overlay'}
                        onClick={() => setLayout('overlay')}
                    >
                        Superpuesto
                    </button>
                </fieldset>

                <fieldset className="adversarial-explorer-segmented semantic-fieldset-reset" aria-label="Capa visual">
                    <button
                        type="button"
                        className={layer === 'image' ? 'active' : ''}
                        aria-pressed={layer === 'image'}
                        onClick={() => setLayer('image')}
                    >
                        Imagen
                    </button>
                    {sources.supportsExplanations && (
                        <button
                            type="button"
                            className={layer === 'explanation' ? 'active' : ''}
                            aria-pressed={layer === 'explanation'}
                            onClick={() => setLayer('explanation')}
                        >
                            {taskType === 'detection' ? 'Eigen-CAM' : 'Grad-CAM'}
                        </button>
                    )}
                </fieldset>

                {sources.supportsAnnotations && layer === 'image' && (
                    <label className="adversarial-explorer-toggle">
                        <input
                            type="checkbox"
                            checked={showAnnotations}
                            onChange={(event) => setShowAnnotations(event.target.checked)}
                        />
                        <span>Mostrar cajas</span>
                    </label>
                )}

                <label className="adversarial-explorer-toggle">
                    <input
                        type="checkbox"
                        checked={showPerturbation}
                        onChange={(event) => setShowPerturbation(event.target.checked)}
                    />
                    <span>Mostrar perturbación</span>
                </label>

                <fieldset className="adversarial-zoom-controls semantic-fieldset-reset" aria-label="Ampliación sincronizada">
                    <button
                        type="button"
                        onClick={() => changeZoom(zoom - EXPLORER_ZOOM_STEP)}
                        disabled={zoom === EXPLORER_MIN_ZOOM}
                        aria-label="Reducir"
                    >
                        −
                    </button>
                    <label>
                        <span>Zoom</span>
                        <input
                            type="range"
                            min={EXPLORER_MIN_ZOOM}
                            max={EXPLORER_MAX_ZOOM}
                            step={EXPLORER_ZOOM_STEP}
                            value={zoom}
                            onChange={(event) => changeZoom(event.target.value)}
                        />
                        <output>{Math.round(zoom * 100)}%</output>
                    </label>
                    <button
                        type="button"
                        onClick={() => changeZoom(zoom + EXPLORER_ZOOM_STEP)}
                        disabled={zoom === EXPLORER_MAX_ZOOM}
                        aria-label="Ampliar"
                    >
                        +
                    </button>
                    <button type="button" onClick={resetView}>
                        Restablecer
                    </button>
                </fieldset>
            </div>

            {layout === 'parallel' ? (
                <div className="adversarial-explorer-grid">
                    <ExplorerImage
                        title="Original"
                        imageUrl={pair.original}
                        alt={`Vista original de ${filename || 'la imagen'}`}
                        zoom={zoom}
                        pan={pan}
                        {...viewportHandlers}
                    />
                    <ExplorerImage
                        title="Tras el ataque"
                        imageUrl={pair.adversarial}
                        alt={`Vista tras el ataque de ${filename || 'la imagen'}`}
                        zoom={zoom}
                        pan={pan}
                        {...viewportHandlers}
                    />
                </div>
            ) : (
                <OverlayComparison
                    originalUrl={pair.original}
                    adversarialUrl={pair.adversarial}
                    zoom={zoom}
                    pan={pan}
                    split={split}
                    viewportHandlers={{
                        ...viewportHandlers,
                        onSplitChange: (event) => setSplit(Number(event.target.value)),
                    }}
                />
            )}

            <p className="adversarial-explorer-help">
                Con zoom activo, arrastra cualquiera de las vistas para desplazar ambas.
                También puedes usar las flechas, +, − y 0 con el visor enfocado.
            </p>

            {showPerturbation && sources.perturbationUrl && (
                <section className="adversarial-perturbation-panel" aria-label="Mapa de perturbación">
                    <div className="adversarial-perturbation-preview">
                        <img
                            src={resolveApiUrl(sources.perturbationUrl)}
                            alt={`Perturbación aplicada a ${filename || 'la imagen'}`}
                        />
                    </div>
                    <div className="adversarial-perturbation-copy">
                        <span className="evaluation-eyebrow">Perturbación</span>
                        <h4>Mapa de perturbación</h4>
                        <p>
                            Representación visual amplificada. Las métricas conservan
                            la magnitud numérica real del ataque.
                        </p>
                        {perturbationMetrics && (
                            <dl className="adversarial-perturbation-metrics">
                                <div>
                                    <dt>L∞</dt>
                                    <dd>{perturbationMetrics.linf ?? '—'}</dd>
                                </div>
                                <div>
                                    <dt>L2</dt>
                                    <dd>{perturbationMetrics.l2 ?? '—'}</dd>
                                </div>
                            </dl>
                        )}
                    </div>
                </section>
            )}
        </section>
    )
}

export default AdversarialImageExplorer
