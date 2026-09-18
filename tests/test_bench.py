import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "bench", Path(__file__).resolve().parents[1] / "benchmarks" / "bench.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def test_payload_uses_fixed_output_and_synthetic_prompt(self):
        payload = bench.make_payload("example-model", 128, 12)
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["messages"][0]["content"].count("alpha "), 12)
        self.assertTrue(payload["ignore_eos"])
        self.assertTrue(payload["stream_options"]["include_usage"])

    def test_median_ignores_missing_values(self):
        self.assertEqual(bench.median([{"v": 1}, {"v": None}, {"v": 3}], "v"), 2)
        self.assertIsNone(bench.median([{}], "v"))


if __name__ == "__main__":
    unittest.main()
