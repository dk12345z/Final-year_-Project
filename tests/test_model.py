import torch

from app.data import CLASS_NAMES
from app.model import HybridClassifier


def test_model_forward_shape() -> None:
    model = HybridClassifier(num_classes=len(CLASS_NAMES), handcrafted_dim=1821, pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, 1821))
    assert logits.shape == (2, len(CLASS_NAMES))
