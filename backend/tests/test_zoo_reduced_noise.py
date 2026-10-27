"""Regresiones: no convertir la imagen 224x224 en un reescalado 32x32."""

import numpy as np
import pytest

import ml.attacks.zoo_reduced_noise as module
from ml.attacks.zoo import ZerothOrderOptimizationAttack
from ml.attacks.zoo_reduced_noise import ReducedNoiseZooAttack
from schemas.attack import AttackParameters


class FakeClassifier:
    input_shape = (3, 224, 224)
    nb_classes = 2

    def __init__(self):
        self.seen = []

    def predict(self, x, batch_size=1):
        assert batch_size == 1
        self.seen.extend(np.asarray(x).copy())
        avg = np.mean(x, axis=(1, 2, 3))
        return np.stack((0.4 - avg, avg), axis=1).astype(np.float32)


class VirtualClassifier:
    def __init__(self, predict_fn, input_shape, nb_classes, clip_values):
        self.predict_fn = predict_fn
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        assert clip_values is None  # Permitir perturbaciones negativas.

    def predict(self, x, batch_size=1):
        return self.predict_fn(x)


class FakeART:
    constructed = []
    returned_noise = 0.0

    def __init__(self, **kwargs):
        assert kwargs["use_resize"] is False
        assert kwargs["batch_size"] == 1
        assert kwargs["nb_parallel"] == 128
        self.estimator = kwargs["classifier"]
        self.constructed.append(self)

    def generate(self, x, y=None):
        assert x.shape == (1, 3, 32, 32)
        np.testing.assert_array_equal(x, np.zeros_like(x))
        self.estimator.predict(x)
        if y is not None:
            assert len(y) == 1
        result = np.full_like(x, self.returned_noise)
        self.estimator.predict(result)
        return result


@pytest.fixture
def fake_art(monkeypatch):
    FakeART.constructed = []
    FakeART.returned_noise = 0.0
    monkeypatch.setattr(module, "ARTZooAttack", FakeART)
    monkeypatch.setattr(module, "BlackBoxClassifier", VirtualClassifier)
    return FakeART


def make_attack(estimator):
    attack = ZerothOrderOptimizationAttack()
    parameters = attack.resolve_parameters(AttackParameters(use_resize=True))
    return ReducedNoiseZooAttack(estimator, parameters)


def test_zero_noise_preserves_every_pixel_and_prediction(fake_art):
    estimator = FakeClassifier()
    x = np.random.default_rng(23).uniform(0.1, 0.9, (1, 3, 224, 224)).astype(np.float32)
    result = make_attack(estimator).generate(x, y=np.array([0]))
    assert np.array_equal(result, x)
    assert np.array_equal(estimator.seen[0], x[0])
    assert np.array_equal(estimator.seen[-1], x[0])


def test_signed_noise_is_added_to_original_not_rescaled_image(fake_art):
    fake_art.returned_noise = -0.08
    estimator = FakeClassifier()
    x = np.random.default_rng(30).uniform(0.2, 0.8, (1, 3, 224, 224)).astype(np.float32)
    result = make_attack(estimator).generate(x, y=np.array([0]))
    np.testing.assert_allclose(result, x - np.float32(0.08), atol=1e-6)
    assert np.array_equal(estimator.seen[-1], result[0])
    assert np.max(np.abs(result - x)) < 0.081


def test_art_state_is_isolated_between_images_and_calls(fake_art):
    estimator = FakeClassifier()
    attack = make_attack(estimator)
    x = np.full((2, 3, 224, 224), 0.5, dtype=np.float32)
    first = attack.generate(x, y=np.array([0, 0]))
    second = attack.generate(x, y=np.array([0, 0]))
    assert np.array_equal(first, x)
    assert np.array_equal(second, x)
    assert len(fake_art.constructed) == 4
    assert len({id(a) for a in fake_art.constructed}) == 4


def test_invalid_label_batch_rejected(fake_art):
    estimator = FakeClassifier()
    x = np.full((2, 3, 224, 224), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="labels"):
        make_attack(estimator).generate(x, y=np.array([0]))


def test_output_clips_to_valid_pixel_range(fake_art):
    fake_art.returned_noise = 0.2
    x = np.ones((1, 3, 224, 224), dtype=np.float32)
    result = make_attack(FakeClassifier()).generate(x)
    assert np.array_equal(result, x)


def test_real_art_zoo_accepts_virtual_classifier_and_returns_valid_tensor():
    """Smoke ligero con ART real: valida el acoplamiento, no exige convergencia."""

    class ToyClassifier:
        input_shape = (3, 8, 8)
        nb_classes = 2

        def predict(self, x, batch_size=1):
            avg = np.mean(x, axis=(1, 2, 3))
            return np.stack((0.5 - avg, avg), axis=1).astype(np.float32)

    attack = ZerothOrderOptimizationAttack()
    parameters = attack.resolve_parameters(
        AttackParameters(use_resize=True, max_iter=2, nb_parallel=4)
    )
    original = np.full((1, 3, 8, 8), 0.2, dtype=np.float32)
    result = ReducedNoiseZooAttack(ToyClassifier(), parameters).generate(
        original, y=np.array([0])
    )
    assert result.shape == original.shape
    assert np.isfinite(result).all()
    assert (0 <= result).all() and (result <= 1).all()


def test_reduced_zoo_exposes_common_art_estimator_contract():
    """Verificar la compatibilidad con el contrato común de ART."""
    estimator = FakeClassifier()
    attack = make_attack(estimator)

    assert attack.estimator is estimator
    assert attack.estimator_requirements == module.ARTZooAttack._estimator_requirements
    assert attack.is_estimator_valid(estimator, attack.estimator_requirements) is False
