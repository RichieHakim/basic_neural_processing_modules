import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _normalize_dep_name(dep: str) -> str:
    name = re.split(r"[<>=!~; ]", dep, maxsplit=1)[0]
    return name.strip().lower().replace("-", "_")


class TestSetupExtras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with PYPROJECT_PATH.open("rb") as f:
            cls.pyproject = tomllib.load(f)
        cls.extras = cls.pyproject["project"]["optional-dependencies"]

    def test_pyproject_exists(self):
        self.assertTrue(PYPROJECT_PATH.exists())

    def test_functional_extras_exist(self):
        expected = {
            "core",
            "plot",
            "ml",
            "io",
            "sql",
            "server",
            "video",
            "gpu",
            "dev",
            "all",
            "all_latest",
        }
        self.assertTrue(expected.issubset(set(self.extras.keys())))

    def test_core_is_lighter_than_legacy_core(self):
        core = {_normalize_dep_name(dep) for dep in self.extras["core"]}
        should_not_be_core = {
            "pytest",
            "hypothesis",
            "paramiko",
            "sqlalchemy",
            "pymysql",
            "wandb",
            "ipywidgets",
        }
        self.assertTrue(core.isdisjoint(should_not_be_core))

    def test_legacy_aliases_still_exist(self):
        aliases = {
            "core_latest",
            "core_cv2Headless",
            "core_latest_cv2Headless",
        }
        self.assertTrue(aliases.issubset(set(self.extras.keys())))

    def test_build_backend_is_setuptools(self):
        build_backend = self.pyproject["build-system"]["build-backend"]
        self.assertEqual(build_backend, "setuptools.build_meta")


if __name__ == "__main__":
    unittest.main()
