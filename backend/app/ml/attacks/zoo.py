from art.attacks.evasion import ZooAttack as ARTZooAttack

from ml.attacks.art_classification import BaseARTClassificationAttack
from ml.attacks.base import AttackParameterSpec
from ml.attacks.zoo_reduced_noise import ReducedNoiseZooAttack


class ZerothOrderOptimizationAttack(BaseARTClassificationAttack):
    id = "zoo"
    name = "Zeroth-Order Optimization (ZOO)"
    description = (
        "Ataque de caja negra basado en puntuaciones que aproxima gradientes mediante "
        "diferencias finitas y optimización coordinada ADAM"
    )
    knowledge_type = "black_box"
    revision = "art-1.20.1-zoo-noise-v2"
    supported_task_types = ("classification",)
    parameter_specs = (
        AttackParameterSpec(
            key="confidence",
            label="Confianza (κ)",
            summary_label="κ",
            description=(
                "Margen de confianza exigido al ejemplo adversario; valores mayores "
                "suelen aumentar la perturbación"
            ),
            value_type="float",
            default=0.0,
            control="number",
            minimum=0.0,
            step=0.1,
        ),
        AttackParameterSpec(
            key="learning_rate",
            label="Tasa de aprendizaje",
            summary_label="tasa",
            description="Tasa de aprendizaje de la optimización coordinada ADAM",
            value_type="float",
            default=0.01,
            control="number",
            minimum=0.000001,
            step=0.001,
        ),
        AttackParameterSpec(
            key="max_iter",
            label="Iteraciones máximas",
            summary_label="iter",
            description="Número máximo de iteraciones por búsqueda del coeficiente c",
            value_type="integer",
            default=10,
            control="number",
            minimum=1,
            maximum=20000,
            step=1,
        ),
        AttackParameterSpec(
            key="binary_search_steps",
            label="Búsquedas binarias",
            summary_label="búsq.",
            description=(
                "Número de ajustes del equilibrio entre distancia y éxito; multiplica "
                "el coste de la optimización"
            ),
            value_type="integer",
            default=1,
            control="number",
            minimum=1,
            maximum=10,
            step=1,
        ),
        AttackParameterSpec(
            key="initial_const",
            label="Constante inicial",
            summary_label="c₀",
            description=(
                "Valor inicial del coeficiente que equilibra distancia y objetivo "
                "adversario"
            ),
            value_type="float",
            default=0.001,
            control="number",
            minimum=0.000000001,
            step=0.001,
        ),
        AttackParameterSpec(
            key="nb_parallel",
            label="Coordenadas paralelas",
            summary_label="coord.",
            description=(
                "Número de coordenadas actualizadas por iteración; cada coordenada "
                "requiere dos evaluaciones para estimar su derivada"
            ),
            value_type="integer",
            default=128,
            control="number",
            minimum=1,
            step=1,
        ),
        AttackParameterSpec(
            key="use_resize",
            label="Usar reducción dimensional",
            summary_label="redim.",
            description=(
                "Optimiza ruido de 32×32 sobre la imagen original, "
                "sin reducir ni degradar la imagen que recibe el modelo. "
                "La búsqueda puede requerir muchas consultas."
            ),
            value_type="boolean",
            default=True,
            control="checkbox",
        ),
        AttackParameterSpec(
            key="variable_h",
            label="Paso de diferencia finita",
            summary_label="h",
            description=(
                "Paso usado para estimar numéricamente las derivadas de cada coordenada"
            ),
            value_type="float",
            default=0.0001,
            control="number",
            minimum=0.000000001,
            step=0.0001,
        ),
    )

    def build_art_attack(self, estimator, parameters):
        if parameters.use_resize:
            return ReducedNoiseZooAttack(estimator=estimator, parameters=parameters)

        return ARTZooAttack(
            classifier=estimator,
            confidence=parameters.confidence,
            targeted=False,
            learning_rate=parameters.learning_rate,
            max_iter=parameters.max_iter,
            binary_search_steps=parameters.binary_search_steps,
            initial_const=parameters.initial_const,
            abort_early=True,
            use_resize=parameters.use_resize,
            use_importance=True,
            nb_parallel=parameters.nb_parallel,
            batch_size=1,
            variable_h=parameters.variable_h,
            verbose=False,
        )
