import os
import subprocess
import sys
import unittest


class TestLazyImports(unittest.TestCase):
    def test_import_bnpm_from_subprocess(self):
        result = subprocess.run(
            [sys.executable, "-c", "import bnpm; print(bnpm.__version__)"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_submodule_not_loaded_until_accessed(self):
        import bnpm

        self.assertNotIn("bnpm.cupy_helpers", sys.modules)
        _ = bnpm.cupy_helpers
        self.assertIn("bnpm.cupy_helpers", sys.modules)

    def test_unknown_attribute_raises_attribute_error(self):
        import bnpm

        with self.assertRaises(AttributeError):
            _ = bnpm.this_attribute_does_not_exist


if __name__ == "__main__":
    unittest.main()
