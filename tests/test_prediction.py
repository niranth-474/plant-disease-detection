import unittest

from PIL import Image

from backend.services.prediction_service import PredictionService
from model.config import BEST_MODEL_PATH, CLASS_NAMES_PATH


class PredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BEST_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
            raise unittest.SkipTest("Trained checkpoint files are not available.")
        cls.service = PredictionService.from_files()

    def test_checkpoint_output_matches_class_names(self):
        self.assertEqual(
            self.service.model.classifier[-1].out_features,
            len(self.service.class_names),
        )

    def test_prediction_contains_three_valid_classes(self):
        result = self.service.predict(Image.new("RGB", (64, 64), "green"))
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertEqual(
            result["predicted_class"],
            result["top_predictions"][0]["class_name"],
        )
        self.assertTrue(
            all(item["class_name"] in self.service.class_names for item in result["top_predictions"])
        )


if __name__ == "__main__":
    unittest.main()
