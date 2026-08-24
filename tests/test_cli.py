import io
import json
import unittest
from contextlib import redirect_stdout

import app


class CommandLineTests(unittest.TestCase):
    def test_json_output_is_valid(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = app.main(["Gym 2 times and sleep 8 hours", "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            sum(day.count("Gym session") for day in payload["days"].values()),
            2,
        )
        self.assertIn(payload["workload"]["level"], {"low", "moderate", "high"})


if __name__ == "__main__":
    unittest.main()
