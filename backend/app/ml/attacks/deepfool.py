from art.attacks.evasion import DeepFool as ARTDeepFool

from ml.attacks.art_classification import BaseARTClassificationAttack
from ml.attacks.base import AttackParameterSpec


class DeepFoolAttack(BaseARTClassificationAttack):
    id = "deepfool"
    name = "DeepFool"
    description = (
        "Ataque de caja blanca que aproxima iterativamente la frontera de decisión "
        "para encontrar perturbaciones pequeñas en clasificación"
    )
    knowledge_type = "white_box"
    revision = "art-1.20.1-deepfool-v1"
    supported_task_types = ("classification",)
    parameter_specs = (
        AttackParameterSpec(
            key="max_iter",
            label="Iteraciones máximas",
            summary_label="iter",
            description=(
                "Número máximo de iteraciones; DeepFool termina antes cuando "
                "consigue cambiar la predicción"
            ),
            value_type="integer",
            default=100,
            control="number",
            minimum=1,
            maximum=1000,
            step=1,
        ),
        AttackParameterSpec(
            key="overshoot",
            label="Factor de sobrepaso (overshoot)",
            summary_label="sobrepaso",
            description=(
                "Factor adicional aplicado a la perturbación encontrada para "
                "sobrepasar la frontera de decisión"
            ),
            value_type="float",
            default=1e-6,
            control="number",
            minimum=0.0,
            step=1e-6,
        ),
        AttackParameterSpec(
            key="nb_grads",
            label="Gradientes de clase",
            summary_label="grad.",
            description=(
                "Número de clases con mayor puntuación cuyos gradientes se "
                "consideran; valores menores reducen el coste por iteración"
            ),
            value_type="integer",
            default=10,
            control="number",
            minimum=1,
            step=1,
        ),
    )

    def build_art_attack(self, estimator, parameters):
        return ARTDeepFool(
            classifier=estimator,
            max_iter=parameters.max_iter,
            epsilon=parameters.overshoot,
            nb_grads=parameters.nb_grads,
            batch_size=1,
            verbose=False,
        )
