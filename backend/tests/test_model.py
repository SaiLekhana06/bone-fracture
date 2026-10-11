import pytest
import torch
from pathlib import Path
from backend.ml.model import build_model, load_and_validate_checkpoint

def test_model_build():
    model = build_model(num_classes=2, pretrained=False)
    assert model is not None
    assert model.num_classes == 2

    # Verify forward pass shape
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)

def test_model_gradcam_target():
    model = build_model(num_classes=2, pretrained=False)
    target_layer = model.get_gradcam_target_layer()
    assert target_layer is not None

def test_checkpoint_nonexistent():
    fake_path = Path("/tmp/nonexistent_model_checkpoint.pt")
    model, meta, err = load_and_validate_checkpoint(fake_path, num_classes=2)
    assert model is None
    assert "not found" in err.lower()
