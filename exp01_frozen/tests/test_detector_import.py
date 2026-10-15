"""Exercise the launcher with a stand-in detector; never imports torch or trains."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


DETECTOR = Path(__file__).resolve().parents[1] / "detector.py"


class DetectorImportTest(unittest.TestCase):
    def test_child_can_import_local_repo_and_existing_pythonpath(self):
        with tempfile.TemporaryDirectory(prefix="ccs_si_import_") as tmp:
            root = Path(tmp)
            repo = root / "local detector repo"
            package = repo / "ultralytics"
            package.mkdir(parents=True)
            extra = root / "existing_pythonpath"
            extra.mkdir()
            (extra / "preserved_path_token.py").write_text("TOKEN = 'existing-path-preserved'\n")
            (repo / "torch.py").write_text("class cuda:\n    @staticmethod\n    def is_available():\n        return False\n")
            (package / "__init__.py").write_text(textwrap.dedent('''\
                import subprocess
                import sys
                from pathlib import Path

                class YOLO:
                    def __init__(self, model):
                        self.cwd = Path(model).parent

                    def train(self, **kwargs):
                        command = "import ultralytics, preserved_path_token; print('DDP_IMPORT_OK', ultralytics.__file__, preserved_path_token.TOKEN)"
                        subprocess.run([sys.executable, "-c", command], cwd=self.cwd, check=True)
                '''))
            run = root / "run"
            run.mkdir()
            cfg = {"detector_repo": str(repo), "imgsz": 64, "seed": 0}
            (run / "config.json").write_text(json.dumps(cfg))
            (run / "manifest.jsonl").write_text("")
            (run / "detector_arch.yaml").write_text("nc: 10\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(extra)
            result = subprocess.run(
                [sys.executable, str(DETECTOR), "train", "--run", str(run), "--device", "cpu"],
                cwd=root, env=env, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("DDP_IMPORT_OK", result.stdout)
            self.assertIn(str(package / "__init__.py"), result.stdout)
            self.assertIn("existing-path-preserved", result.stdout)


if __name__ == "__main__":
    unittest.main()
