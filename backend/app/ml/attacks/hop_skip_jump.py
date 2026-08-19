import numpy as np
from art.attacks.evasion import HopSkipJump as ARTHopSkipJump

from ml.attacks.art_classification import BaseARTClassificationAttack
from ml.attacks.base import AttackParameterChoice, AttackParameterSpec
from schemas.attack import AttackParameters


class HopSkipJumpAttack(BaseARTClassificationAttack):
    id = "hop_skip_jump"
    name = "HopSkipJump"
    description = (
        "Ataque black-box basado en decisiones que aproxima la frontera de "
        "clasificación usando únicamente predicciones finales del modelo"
    )
    knowledge_type = "black_box"
    revision = "art-1.20.1-hop-skip-jump-v1"
    supported_task_types = ("classification",)
    parameter_specs = (
        AttackParameterSpec(
            key="norm",
            label="Norma",
            summary_label="norm",
            description="Norma usada para minimizar la perturbación adversaria",
            value_type="string",
            default="l2",
            control="select",
            choices=(
                AttackParameterChoice(value="l2", label="L2"),
                AttackParameterChoice(value="linf", label="L∞"),
            ),
        ),
        AttackParameterSpec(
            key="max_iter",
            label="Iteraciones máximas",
            summary_label="iter",
            description=(
                "Número máximo de pasos de refinamiento sobre la frontera de decisión"
            ),
            value_type="integer",
            default=50,
            control="number",
            minimum=1,
            maximum=1000,
            step=1,
        ),
        AttackParameterSpec(
            key="max_eval",
            label="Evaluaciones máximas por iteración",
            summary_label="max eval",
            description=(
                "Límite de consultas usadas para estimar la dirección de ataque en "
                "una iteración"
            ),
            value_type="integer",
            default=10000,
            control="number",
            minimum=1,
            maximum=10000,
            step=1,
        ),
        AttackParameterSpec(
            key="init_eval",
            label="Evaluaciones iniciales",
            summary_label="init eval",
            description=(
                "Consultas usadas al inicio para estimar la dirección; el número "
                "crece con las iteraciones hasta max_eval"
            ),
            value_type="integer",
            default=100,
            control="number",
            minimum=1,
            maximum=10000,
            step=1,
        ),
        AttackParameterSpec(
            key="init_size",
            label="Intentos de inicialización",
            summary_label="init",
            description=(
                "Número máximo de muestras aleatorias probadas para encontrar un "
                "ejemplo adversario inicial"
            ),
            value_type="integer",
            default=100,
            control="number",
            minimum=1,
            maximum=10000,
            step=1,
        ),
    )

    def resolve_parameters(self, parameters: AttackParameters) -> AttackParameters:
        resolved = super().resolve_parameters(parameters)
        if resolved.init_eval > resolved.max_eval:
            raise ValueError("Parameter 'init_eval' must be less than or equal to 'max_eval'")
        return resolved

    def build_art_attack(self, estimator, parameters):
        norm = 2 if parameters.norm == "l2" else np.inf
        return ARTHopSkipJump(
            classifier=estimator,
            batch_size=64,
            targeted=False,
            norm=norm,
            max_iter=parameters.max_iter,
            max_eval=parameters.max_eval,
            init_eval=parameters.init_eval,
            init_size=parameters.init_size,
            verbose=False,
        )
