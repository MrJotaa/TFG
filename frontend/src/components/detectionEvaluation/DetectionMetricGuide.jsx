function DetectionMetricGuide() {
    return (
        <details className="detection-metric-guide">
            <summary>
                <span className="detection-info-icon" aria-hidden="true">i</span>
                <span>
                    <strong>Guía de métricas</strong>
                    <small>Definiciones, unidades y sentido de lectura</small>
                </span>
            </summary>
            <div className="detection-metric-guide-grid">
                <section>
                    <h4>COCO: rendimiento global</h4>
                    <p><strong>AP</strong> resume precisión y recall; <strong>mAP</strong> la promedia entre clases e IoU. <strong>AR</strong> mide cuánto ground truth se recupera. En estas métricas, un valor mayor es mejor.</p>
                </section>
                <section>
                    <h4>Punto de operación</h4>
                    <p><strong>Precisión</strong> penaliza falsos positivos, <strong>recall</strong> penaliza objetos omitidos y <strong>F1</strong> equilibra ambos. TP, FP y FN son conteos al umbral de confianza e IoU configurado.</p>
                </section>
                <section>
                    <h4>Comparación adversarial</h4>
                    <p>Los deltas COCO son <strong>original − adversarial</strong>: un valor positivo indica pérdida. Preservado, suprimido, recuperado y no detectado describen la transición de cada objeto anotado.</p>
                </section>
                <section>
                    <h4>TIDE: causa del error</h4>
                    <p><strong>dAP</strong> son los puntos de AP50 que se recuperarían al corregir un tipo de error. El cambio de impacto es <strong>adversarial − original</strong>: positivo significa que ese error pesa más tras el ataque.</p>
                </section>
            </div>
        </details>
    )
}

export default DetectionMetricGuide
