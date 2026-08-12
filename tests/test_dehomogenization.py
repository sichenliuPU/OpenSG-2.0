from pathlib import Path
import tempfile
import unittest

import numpy as np

from fe_jax.dehomogenization import recover_beam_states
from fe_jax.dehomogenization import _junction_solid_source
from fe_jax.hybrid_homogenization import homogenize
from fe_jax.sc_glb_input import GlobalFields, read_global_fields
from fe_jax.sc_hybrid_input import read_hybrid_supplement
from fe_jax.sc_local_output import write_local_outputs
from fe_jax.dehomogenization import LocalFields
from fe_jax.vabs_localization import global_file_text


class TestDehomogenization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = Path(__file__).parents[1] / "examples" / "beam_hybrid"

    def test_global_fields_accept_strain_and_stress(self):
        stiffness = np.diag([10.0, 20.0, 30.0, 4.0, 5.0, 6.0])
        compliance = np.linalg.inv(stiffness)
        strain = np.arange(1.0, 7.0) * 1.0e-3
        with tempfile.TemporaryDirectory() as directory:
            strain_path = Path(directory) / "strain.glb"
            strain_path.write_text(
                "0 0 0\n1 0 0\n0 1 0\n0 0 1\n1\n"
                + " ".join(map(str, strain)) + "\n\n"
            )
            stress_path = Path(directory) / "stress.glb"
            stress_path.write_text(
                "0,0,0\n1,0,0\n0,1,0\n0,0,1\n0\n"
                + ",".join(map(str, stiffness @ strain)) + "\n\n"
            )
            by_strain = read_global_fields(strain_path, stiffness, compliance)
            by_stress = read_global_fields(stress_path, stiffness, compliance)
        np.testing.assert_allclose(by_strain.strain, strain)
        np.testing.assert_allclose(by_strain.stress, stiffness @ strain)
        np.testing.assert_allclose(by_stress.strain, strain)

    def test_optional_beam_recovery_record(self):
        source = (self.examples / "rigid_timoshenko_cross.sc.msg").read_text()
        source += "\nBEAM_RECOVERY 1 section.sg\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sc.msg"
            path.write_text(source)
            supplement = read_hybrid_supplement(path)
        self.assertEqual(supplement.beam_recovery[1].source.name, "section.sg")

    def test_vabs_stress_resultant_order(self):
        lines = global_file_text(
            np.zeros(3), np.zeros(3), np.arange(1.0, 7.0)
        ).splitlines()
        np.testing.assert_allclose(
            np.fromstring(lines[4], sep=" "), [1.0, 4.0, 5.0, 6.0]
        )
        np.testing.assert_allclose(
            np.fromstring(lines[5], sep=" "), [2.0, 3.0]
        )

    def test_mode2_requires_same_basename_junction_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stiffness = directory / "junction.sc.kj"
            with self.assertRaisesRegex(FileNotFoundError, "same-basename"):
                _junction_solid_source(stiffness)
            solid = directory / "junction.sc"
            solid.write_text("solid")
            Path(str(solid) + ".msg").write_text("interfaces")
            self.assertEqual(_junction_solid_source(stiffness), solid)

    def test_uniform_extension_beam_state(self):
        path = self.examples / "rigid_timoshenko_cross.sc"
        model, supplement, result = homogenize(path)
        strain = np.array([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
        fields = GlobalFields(
            displacement=np.zeros(3), deformation=np.eye(3), input_flag=1,
            strain=strain, stress=result.effective_stiffness @ strain,
        )
        states = recover_beam_states(result, supplement, fields, np.array([-1.0, 1.0]))
        plus_x = [state for state in states if state.element_id == 1]
        np.testing.assert_allclose(
            [state.generalized_strain[0] for state in plus_x],
            [1.0e-3, 1.0e-3], rtol=1.0e-6, atol=1.0e-12,
        )
        np.testing.assert_allclose(
            [state.resultants[0] for state in plus_x],
            [7000.0, 7000.0], rtol=1.0e-6, atol=1.0e-7,
        )

    def test_swiftcomp_local_output_columns(self):
        fields = LocalFields(
            coordinates=np.array([[1.0, 2.0, 3.0]]),
            displacement=np.array([[4.0, 5.0, 6.0]]),
            strain=np.arange(6.0).reshape(1, 6),
            stress=np.arange(10.0, 16.0).reshape(1, 6),
            region=np.array(["beam"]), owner_id=np.array([7]),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.sc"
            displacement_path, nodal_path = write_local_outputs(source, fields)
            displacement_values = displacement_path.read_text().split()
            nodal_values = nodal_path.read_text().split()
        self.assertEqual(len(displacement_values), 4)
        self.assertEqual(len(nodal_values), 15)
        self.assertEqual(displacement_values[0], "1")


if __name__ == "__main__":
    unittest.main()
