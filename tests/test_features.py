from app.features import FEATURE_SIZE, extract_handcrafted
import numpy as np


def test_handcrafted_feature_shape_and_finiteness() -> None:
    feature_vector = extract_handcrafted(np.zeros((224, 224, 3), dtype=np.uint8))
    assert feature_vector.shape == (FEATURE_SIZE,)
    assert np.isfinite(feature_vector).all()
