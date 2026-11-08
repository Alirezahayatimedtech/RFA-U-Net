from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from choroid_refiner import ChoroidRefiner, NativeRefiner, checkpoint_settings


class ChoroidRefinerTests(unittest.TestCase):
    def test_checkpoint_settings_and_rectangular_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "refiner.safetensors"
            model = ChoroidRefiner(with_prior=False)
            save_file(model.state_dict(), checkpoint, metadata={
                "with_prior": "false", "threshold": "0.55", "flip": "true", "postprocess": "layer",
            })
            with_prior, threshold, flip, postprocess = checkpoint_settings(checkpoint)
            self.assertEqual((with_prior, threshold, flip, postprocess), (False, 0.55, True, "layer"))
            loaded = NativeRefiner(checkpoint, with_prior, torch.device("cpu"))
            probabilities = loaded.predict(Image.new("L", (151, 103), 120), flip=flip)
            self.assertEqual(tuple(probabilities.shape), (103, 151))
            self.assertTrue(torch.isfinite(probabilities).all())

    def test_untrained_residual_preserves_the_native_prior(self):
        refiner = NativeRefiner.__new__(NativeRefiner)
        refiner.model = ChoroidRefiner(with_prior=True).eval()
        refiner.device = torch.device("cpu")
        refiner.prior = None
        logits = torch.linspace(-6, 6, 23 * 37).view(23, 37)
        scan = Image.fromarray(np.full((23, 37), 120, dtype=np.uint8))
        torch.testing.assert_close(refiner.predict(scan, flip=True, existing_prior=logits), logits.sigmoid())


if __name__ == "__main__":
    unittest.main()
