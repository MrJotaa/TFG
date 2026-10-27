"""ZOO sobre perturbaciones de baja resolución, sin reescalar la imagen fuente.

ART 1.20.1, con ``use_resize=True``, optimiza una copia reducida de la
imagen y puede devolverla ampliada incluso cuando el optimizador no añade ruido.
Esta adaptación conserva el optimizador oficial de ART, pero cambia su dominio:
la entrada virtual de ZOO es un mapa de perturbación (inicialmente cero).
El clasificador virtual evalúa siempre ``clip(imagen_original + upsample(ruido))``.

La distancia del objetivo interno de ART sigue midiéndose sobre el mapa de
ruido reducido. Las métricas de la plataforma se calculan, como siempre, en
el tensor completo tras la interpolación y el clipping.
"""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import ZooAttack as ARTZooAttack
from art.estimators.classification import BlackBoxClassifier
from scipy.ndimage import zoom


class ReducedNoiseZooAttack:
    """Adaptador con ``generate(x, y)`` compatible con el runner común de ART.

    No retiene estado de optimización por imagen: para cada muestra se crea
    un clasificador virtual y una instancia independiente de ART ZooAttack.
    Se reutilizan el modelo real y su adaptador durante toda la evaluación.
    """

    def __init__(self, estimator, parameters) -> None:
        self._estimator = estimator
        self._parameters = parameters

    @property
    def estimator(self):
        """Exponer el clasificador original mediante el contrato de ART."""
        return self._estimator

    @property
    def estimator_requirements(self):
        """Exponer los requisitos del estimador de ART ZooAttack."""
        return ARTZooAttack._estimator_requirements

    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        return ARTZooAttack.is_estimator_valid(
            estimator, estimator_requirements
        )

    @staticmethod
    def _compose(clean: np.ndarray, reduced_noise: np.ndarray) -> np.ndarray:
        """Aplicar solamente ruido al original; ruido cero => identidad exacta."""
        if clean.ndim != 4 or reduced_noise.ndim != 4:
            raise ValueError("ZOO expects NCHW image batches")
        if clean.shape[:2] != reduced_noise.shape[:2]:
            raise ValueError("ZOO noise must match the original batch and channels")
        if reduced_noise.shape[2:] == clean.shape[2:]:
            expanded = reduced_noise
        else:
            expanded = zoom(
                reduced_noise,
                (
                    1,
                    1,
                    clean.shape[2] / reduced_noise.shape[2],
                    clean.shape[3] / reduced_noise.shape[3],
                ),
                order=1,
                mode="nearest",
            )
        # float32 no-op for zero noise. Bounds are the platform attack domain.
        return np.clip(clean + expanded, 0.0, 1.0).astype(np.float32)

    def generate(
        self, x: np.ndarray, y: np.ndarray | None = None, **kwargs
    ) -> np.ndarray:
        del kwargs
        images = np.asarray(x, dtype=np.float32)
        if images.ndim != 4 or images.shape[1] != self._estimator.input_shape[0]:
            raise ValueError("Reduced-noise ZOO requires NCHW classification inputs")
        if not np.isfinite(images).all():
            raise ValueError("ZOO input must contain finite values")
        if y is not None and len(y) != len(images):
            raise ValueError("ZOO labels must match the image batch length")

        results = []
        for index in range(len(images)):
            clean = np.ascontiguousarray(images[index : index + 1])
            height = min(32, clean.shape[2])
            width = min(32, clean.shape[3])
            noise_shape = (clean.shape[1], height, width)
            initial_noise = np.zeros((1, *noise_shape), dtype=np.float32)

            def predict_noise(noise: np.ndarray) -> np.ndarray:
                # La imagen limpia es siempre la referencia, no su versión 32x32.
                expanded_clean = np.broadcast_to(clean, (len(noise), *clean.shape[1:]))
                candidate = self._compose(expanded_clean, noise)
                return self._estimator.predict(candidate, batch_size=1)

            virtual_classifier = BlackBoxClassifier(
                predict_fn=predict_noise,
                input_shape=noise_shape,
                nb_classes=self._estimator.nb_classes,
                # ¡Crucial! El ruido necesita valores positivos Y negativos.
                clip_values=None,
            )
            parameters = self._parameters
            optimizer = ARTZooAttack(
                classifier=virtual_classifier,
                confidence=parameters.confidence,
                targeted=False,
                learning_rate=parameters.learning_rate,
                max_iter=parameters.max_iter,
                binary_search_steps=parameters.binary_search_steps,
                initial_const=parameters.initial_const,
                abort_early=True,
                use_resize=False,  # No permitir que ART reduzca la imagen.
                use_importance=True,
                nb_parallel=min(parameters.nb_parallel, initial_noise.size),
                batch_size=1,
                variable_h=parameters.variable_h,
                verbose=False,
            )
            sample_y = None if y is None else np.asarray(y)[index : index + 1]
            resulting_noise = optimizer.generate(x=initial_noise, y=sample_y)
            results.append(self._compose(clean, resulting_noise))

        return np.concatenate(results, axis=0)
