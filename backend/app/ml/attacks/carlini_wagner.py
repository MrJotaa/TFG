from art.attacks.evasion import CarliniL2Method, CarliniLInfMethod

from ml.attacks.art_classification import BaseARTClassificationAttack
from ml.attacks.base import AttackParameterChoice, AttackParameterSpec


class CarliniWagnerAttack(BaseARTClassificationAttack):
    id = "carlini_wagner"
    name = "Carlini & Wagner"
    description = (
        "Ataque white-box de optimización C&W para clasificación; "
        "coste computacional alto"
    )
    knowledge_type = "white_box"
    revision = "art-1.20.1-cw-l2-linf-v2"
    supported_task_types = ("classification",)
    parameter_specs = (
        AttackParameterSpec(
            key="norm",
            label="Norma",
            summary_label="norm",
            description="Norma que minimiza la variante de Carlini & Wagner",
            value_type="string",
            default="l2",
            control="select",
            choices=(
                AttackParameterChoice(value="l2", label="L2"),
                AttackParameterChoice(value="linf", label="L∞"),
            ),
        ),
        AttackParameterSpec(
            key="confidence",
            label="Confidence",
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
            label="Learning rate",
            summary_label="lr",
            description="Tasa de aprendizaje de la optimización C&W",
            value_type="float",
            default=0.01,
            control="number",
            minimum=0.000001,
            step=0.001,
        ),
        AttackParameterSpec(
            key="binary_search_steps",
            label="Búsquedas binarias (L2)",
            summary_label="bss",
            description=(
                "Número de ajustes del equilibrio distancia/confianza en C&W L2. "
                "Es el principal multiplicador de coste; no se usa en L∞."
            ),
            value_type="integer",
            default=3,
            control="number",
            minimum=1,
            maximum=10,
            step=1,
        ),
        AttackParameterSpec(
            key="max_iter",
            label="Iteraciones máximas",
            summary_label="iter",
            description="Número máximo de pasos de optimización por búsqueda interna",
            value_type="integer",
            default=10,
            control="number",
            minimum=1,
            maximum=1000,
            step=1,
        ),
    )

    def resolve_parameters(self, parameters):
        resolved = super().resolve_parameters(parameters)
        if resolved.norm == "l2":
            return resolved

        return type(resolved)(
            norm=resolved.norm,
            confidence=resolved.confidence,
            learning_rate=resolved.learning_rate,
            max_iter=resolved.max_iter,
        )

    def build_art_attack(self, estimator, parameters):
        common_parameters = {
            "classifier": estimator,
            "confidence": parameters.confidence,
            "targeted": False,
            "learning_rate": parameters.learning_rate,
            "max_iter": parameters.max_iter,
            "batch_size": 1,
            "verbose": False,
        }

        if parameters.norm == "l2":
            return CarliniL2Method(
                **common_parameters,
                binary_search_steps=parameters.binary_search_steps,
                initial_const=0.01,
                max_halving=5,
                max_doubling=5,
            )

        if parameters.norm == "linf":
            return CarliniLInfMethod(
                **common_parameters,
                decrease_factor=0.9,
                initial_const=1e-5,
                largest_const=20.0,
                const_factor=2.0,
            )

        raise ValueError(f"Unsupported Carlini & Wagner norm '{parameters.norm}'")
