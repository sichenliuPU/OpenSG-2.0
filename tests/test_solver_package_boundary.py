from pathlib import Path
import tomllib
import unittest


class TestSolverPackageBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]

    def test_wheel_contains_only_solver_package(self):
        configuration = tomllib.loads(
            (self.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        include = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["include"]
        self.assertEqual(include, ["fe_jax"])

    def test_mesh_and_plot_dependencies_are_optional(self):
        configuration = tomllib.loads(
            (self.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        required = {
            value.split(">=")[0].lower()
            for value in configuration["project"]["dependencies"]
        }
        self.assertNotIn("gmsh", required)
        self.assertNotIn("matplotlib", required)
        optional = configuration["project"]["optional-dependencies"]
        self.assertIn("gmsh", optional["input-builders"])
        self.assertIn("matplotlib", optional["examples"])

    def test_solver_modules_do_not_reference_examples_or_builders(self):
        forbidden = (
            "examples",
            "tests",
            "tools.",
            "gmsh",
            "matplotlib",
            "build_boolean_junction",
            "build_simple_cubic",
        )
        for name in ("hybrid_homogenization.py", "dehomogenization.py", "opensg.py"):
            source = (self.root / "fe_jax" / name).read_text(encoding="utf-8").lower()
            for value in forbidden:
                self.assertNotIn(value, source, f"{name} references {value}")

    def test_package_import_does_not_load_plotting_helpers(self):
        for name in ("__init__.py", "utils.py"):
            source = (self.root / "fe_jax" / name).read_text(encoding="utf-8")
            self.assertNotIn("from .profiling import", source)

        profiling = (self.root / "fe_jax" / "profiling.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("\nimport matplotlib", profiling)

    def test_example_element_builders_are_not_installed(self):
        self.assertFalse((self.root / "fe_jax" / "junction_boolean.py").exists())
        self.assertFalse((self.root / "fe_jax" / "junction_c3d20.py").exists())
        self.assertTrue(
            (self.root / "examples" / "simple_cubic_dehomogenization"
             / "c3d20_recovery.py").is_file()
        )
        self.assertTrue((self.root / "tools" / "junction_boolean.py").is_file())

    def test_simple_cubic_example_is_self_contained(self):
        example = self.root / "examples" / "simple_cubic_dehomogenization"
        source = (example / "verify_dehomogenization.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from fe_jax import LocalFields, dehomogenize", source)
        self.assertNotIn("recover_beam_states", source)
        self.assertNotIn("recover_vabs_fields", source)
        self.assertNotIn("parents[3]", source)
        for path in (
            example / "inputs" / "square0.sg",
            example / "inputs" / "simple_cubic.glb",
            example
            / "reference"
            / "cross_nSG3_3D_C3D20Rpbc_nodal_stress.csv",
        ):
            self.assertTrue(path.is_file(), f"Missing example input: {path}")


if __name__ == "__main__":
    unittest.main()
