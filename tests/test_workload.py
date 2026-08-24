import unittest

from tools.workload_model import heuristic_predict


class WorkloadPredictorTests(unittest.TestCase):
    def test_busy_demo_week_is_not_classified_as_light(self):
        prediction = heuristic_predict([8.5, 7.5, 0, 1, 7])
        self.assertEqual(prediction, 2)


if __name__ == "__main__":
    unittest.main()
