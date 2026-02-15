import os
import subprocess
import sys
import unittest
from pathlib import Path
import ast


class TestPhase2OptionalImports(unittest.TestCase):
    def test_import_sql_and_server_modules_from_subprocess(self):
        code = (
            "import bnpm\n"
            "import bnpm.sql_helpers\n"
            "import bnpm.server\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_sql_helper_pure_url_functions_do_not_require_sqlalchemy(self):
        import bnpm.sql_helpers as sql_helpers

        url = sql_helpers.make_url(
            sql_type="postgresql",
            username="alice",
            password="secret",
            host="db.example.com",
            port=5432,
            database="analytics",
        )
        self.assertEqual(
            url,
            "postgresql+psycopg2://alice:secret@db.example.com:5432/analytics",
        )

        parsed = sql_helpers.parse_url(url)
        self.assertEqual(parsed["sql_type"], "postgresql")
        self.assertEqual(parsed["username"], "alice")
        self.assertIn("database", parsed)

    def test_sql_runtime_dependency_error_is_deferred_until_needed(self):
        import bnpm.sql_helpers as sql_helpers

        try:
            connection = sql_helpers.make_sql_connection("sqlite:///tmp.db")
        except ImportError:
            connection = None

        if connection is not None:
            connection.close()

    def test_time_series_no_top_level_jax_or_pandas_import(self):
        path_time_series = Path(__file__).resolve().parents[1] / "timeSeries.py"
        tree = ast.parse(path_time_series.read_text())
        top_level_imports = set()

        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                top_level_imports.add(node.module.split(".")[0])

        self.assertNotIn("jax", top_level_imports)
        self.assertNotIn("pandas", top_level_imports)


if __name__ == "__main__":
    unittest.main()
