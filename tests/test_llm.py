import unittest

from core.llm import LLMError, extract_json_object


class JsonExtractionTests(unittest.TestCase):
    def test_extracts_json_from_extra_model_text(self):
        result = extract_json_object('Here is the result:\n```json\n{"ok": true}\n```')
        self.assertEqual(result, {"ok": True})

    def test_rejects_output_without_an_object(self):
        with self.assertRaises(LLMError):
            extract_json_object("no structured response")


if __name__ == "__main__":
    unittest.main()
