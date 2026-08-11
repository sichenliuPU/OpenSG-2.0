from pathlib import Path
import tempfile
import unittest

import numpy as np

from fe_jax.beam import beam_frame
from fe_jax.dehomogenization import recover_beam_states
from fe_jax.dehomogenization import _junction_solid_source
from fe_jax.hybrid_homogenization import homogenize
from fe_jax.junction import JunctionConnectionPoint
from fe_jax.junction_boolean import (
    build_boolean_junction,
    write_solid_junction_input,
)
from fe_jax.junction_c3d20 import build_simple_cubic_c3d20_mesh
from fe_jax.junction_solid import analyze_junction
from fe_jax.sc_glb_input import GlobalFields, read_global_fields
from fe_jax.sc_hybrid_input import read_hybrid_supplement, read_solid_junction
from fe_jax.sc_local_output import write_local_outputs
from fe_jax.dehomogenization import LocalFields


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

    def test_boolean_junction_linear_and_quadratic(self):
        directions = np.vstack((np.eye(3), -np.eye(3)))
        connection_points = []
        for identifier, direction in enumerate(directions, start=1):
            trial = (
                np.array([0.0, 0.0, 1.0])
                if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            )
            frame = beam_frame(np.zeros(3), direction, trial)
            connection_points.append(JunctionConnectionPoint(
                identifier, 0.015 * direction, frame
            ))
        for element_flag, number_of_nodes in ((0, 4), (1, 10)):
            model = build_boolean_junction(
                tuple(connection_points), "square", 0.005, 0.006,
                element_flag, 1, (70.0e9, 0.3),
            )
            self.assertTrue(all(len(element) == number_of_nodes for element in model.elements))
            solution = analyze_junction(model)
            self.assertEqual(solution.stiffness.matrix.shape, (36, 36))
            self.assertEqual(solution.displacement_recovery.shape[1], 36)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "junction.sc"
                write_solid_junction_input(path, model)
                reloaded = read_solid_junction(path)
            self.assertEqual(len(reloaded.elements), len(model.elements))
            self.assertEqual(
                len(reloaded.connection_points), len(model.connection_points)
            )

    def test_c3d20_simple_cubic_has_exact_centerline_nodes(self):
        mesh = build_simple_cubic_c3d20_mesh(
            side=0.01, stub_length=0.01, elements_per_side=4
        )
        tolerance = 1.0e-12
        mask = (
            (mesh.nodes[:, 0] >= -tolerance)
            & (mesh.nodes[:, 0] <= 0.015 + tolerance)
            & (np.abs(mesh.nodes[:, 1]) <= tolerance)
            & (np.abs(mesh.nodes[:, 2]) <= tolerance)
        )
        centerline = np.sort(mesh.nodes[mask, 0])
        np.testing.assert_allclose(
            centerline, np.arange(0.0, 0.015 + 0.000625, 0.00125),
            rtol=0.0, atol=tolerance,
        )
        self.assertEqual(mesh.elements.shape[1], 20)

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
