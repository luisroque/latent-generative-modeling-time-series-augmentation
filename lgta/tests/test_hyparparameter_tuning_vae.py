import unittest
from lgta.model.hyperparameter_tuning import optimize_hyperparameters


class TestModel(unittest.TestCase):
    def test_hyperparameter_tuning_returns_result(self):
        result = optimize_hyperparameters(
            dataset_name="synthetic", freq="D", n_calls=5, max_epochs=2
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.x)
        self.assertIsNotNone(result.fun)
