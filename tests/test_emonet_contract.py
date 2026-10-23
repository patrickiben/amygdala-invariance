"""Small uncopyrighted fixtures catch the corrected extractor's scale/layer regressions."""
import subprocess
import sys
import types
import os

import numpy as np
import pytest

from amyg_inv.encoders import emonet_contract as contract


def test_raw_rgb_pixels_are_not_normalized():
    image = np.zeros((2, 227, 227, 3), dtype=np.uint8)
    image[0, :, :, 0] = 255
    image[1, :, :, 1] = 128
    batch = contract.emonet_pixels(image)
    assert batch.shape == (2, 3, 227, 227)
    assert batch.dtype == np.float32
    assert np.all(batch[0, 0] == 255)
    assert np.all(batch[1, 1] == 128)
    with pytest.raises(ValueError, match="uint8"):
        contract.emonet_pixels(image.astype(np.float32) / 255)


def test_logits_cannot_be_saved_as_penultimate_features():
    assert contract.checked_features(np.ones((2, 4096)), 2, 4096, "EmoNet").shape == (2, 4096)
    with pytest.raises(ValueError, match="4096"):
        contract.checked_features(np.ones((2, 20)), 2, 4096, "EmoNet")
    invalid = np.ones((2, 4096)); invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        contract.checked_features(invalid, 2, 4096, "EmoNet")


def test_model_load_is_pinned_and_avoids_models_module_collision(tmp_path, monkeypatch):
    # Tiny upstream-interface fixture: exercise checkout/import plumbing without torch/weights.
    (tmp_path / "models.py").write_text("class EmoNet:\n    layer = 'conv_6'\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "models.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Fixture"], check=True)
    head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setitem(sys.modules, "models", types.ModuleType("unrelated_models"))
    with pytest.raises(ValueError, match="commit"):
        contract.load_pinned_emonet_class(tmp_path)
    monkeypatch.setattr(contract, "EMONET_COMMIT", head)
    assert contract.load_pinned_emonet_class(tmp_path).layer == "conv_6"
    (tmp_path / "models.py").write_text("class EmoNet:\n    layer = 'logits'\n")
    with pytest.raises(ValueError, match="unmodified"):
        contract.load_pinned_emonet_class(tmp_path)


def test_pinned_upstream_architecture_on_synthetic_pixels():
    """Optional CPU smoke: real upstream architecture, random weights, no film/data."""
    repo = os.environ.get("EMONET_REPO")
    if not repo:
        pytest.skip("Set EMONET_REPO to the pinned checkout for the optional torch architecture check.")
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = contract.load_pinned_emonet_class(repo)().eval()
    captured = {}
    model.conv_6.register_forward_hook(lambda module, args, out: captured.update(features=out.flatten(1).cpu().numpy()))
    rgb = np.zeros((1, 227, 227, 3), dtype=np.uint8)
    rgb[:, :100, :, 0] = 255
    with torch.no_grad():
        logits = model(torch.from_numpy(contract.emonet_pixels(rgb)))
    assert tuple(logits.shape) == (1, 20)
    assert contract.checked_features(captured["features"], 1, 4096, "conv_6").shape == (1, 4096)
