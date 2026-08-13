from pathlib import Path
import inspect
import shutil
import tempfile
import unittest

import numpy as np
from scipy import linalg, sparse

from fe_jax import beam_timoshenko
from fe_jax.beam import (
    BeamTheory,
    BeamType,
    HomogenizationTerms,
    add_element_terms,
    beam_frame,
    create_homogenization_terms,
    periodic_reduction,
)
from fe_jax.beam_euler import element_terms as euler_element_terms, local_stiffness
from fe_jax.hybrid_homogenization import homogenize, solve_homogenization
from fe_jax.junction import (
    JunctionConnection,
    JunctionInstance,
    JunctionConnectionPoint,
    JunctionStiffness,
    JunctionType,
    connection_matrices,
    read_junction_stiffness,
    rigid_connection_modes,
    write_junction_stiffness,
)
from fe_jax.junction_solid import assemble_solid_stiffness, calculate_junction_stiffness
from fe_jax.sc_hybrid_input import (
    HybridSupplement,
    StructuralGenomeInput,
    apply_periodic_ownership,
    build_beam_discretization,
    complete_periodic_connection_shifts,
    read_solid_junction,
    read_structural_genome,
)


class TestBeamHybridHomogenization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = Path(__file__).parents[2] / "examples" / "opensg" / "beam_hybrid"

    def test_timoshenko_module_has_no_alpha(self):
        source = inspect.getsource(beam_timoshenko).lower()
        self.assertNotIn("alpha", source)

    def test_rigid_timoshenko_regression(self):
        path = self.examples / "rigid_timoshenko_cross.sc"
        model, _, result = homogenize(path)
        expected = np.diag(
            [1.75e8, 1.75e8, 1.75e8, 2.170569381265e5,
             2.170569381265e5, 2.170569381265e5]
        )
        self.assertEqual(model.junction_flag, 0)
        self.assertEqual(result.number_of_junctions, 0)
        self.assertEqual(len(model.nodes), 7)
        self.assertEqual(result.number_of_full_dofs, 19 * 6)
        self.assertFalse(result.has_mechanism)
        np.testing.assert_allclose(result.effective_stiffness, expected, rtol=2.0e-11)

    def test_three_node_euler_recovers_two_node_stiffness(self):
        nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        section = np.diag([100.0, 30.0, 40.0, 50.0])
        three_node = euler_element_terms(nodes, np.array([0, 1, 2]), np.eye(3), section)
        endpoint = three_node.e[:12, :12]
        coupling = three_node.e[:12, 12:]
        internal = three_node.e[12:, 12:]
        reduced = endpoint - coupling @ linalg.solve(
            internal, coupling.T, assume_a="sym"
        )
        np.testing.assert_allclose(
            reduced,
            local_stiffness(2.0, section),
            rtol=5.0e-7,
            atol=5.0e-7,
        )

    def test_three_node_euler_matches_mathematica_bcc_result(self):
        original_nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [-5.0, 5.0, 5.0],
                [5.0, -5.0, 5.0],
                [5.0, 5.0, -5.0],
                [-5.0, -5.0, 5.0],
                [-5.0, 5.0, -5.0],
                [5.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0],
            ]
        )
        transverse = np.array(
            [
                [1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
            ]
        ) / np.sqrt(2.0)
        section = np.diag(
            [1.979203372e10, 3.348035592e8, 4.453207587e8, 4.453207587e8]
        )
        nodes = list(original_nodes)
        elements = []
        for end in range(1, 9):
            midpoint = len(nodes)
            nodes.append(0.5 * (original_nodes[0] + original_nodes[end]))
            elements.append((0, end, midpoint))
        nodes = np.asarray(nodes)
        terms = create_homogenization_terms(len(nodes))
        for node_ids, direction in zip(elements, transverse, strict=True):
            frame = beam_frame(nodes[node_ids[0]], nodes[node_ids[1]], direction)
            add_element_terms(
                terms,
                euler_element_terms(nodes, np.asarray(node_ids), frame, section),
            )
        periodic = periodic_reduction(
            len(nodes), [(node, 1) for node in range(2, 9)]
        ).toarray()
        stiffness, _, _ = solve_homogenization(terms, periodic, 1000.0)
        expected = np.array(
            [
                [1.534561321107214e8, 1.5181065386568227e8, 1.5181065386568227e8, 0.0, 0.0, 0.0],
                [1.5181065386568227e8, 1.534561321107214e8, 1.5181065386568227e8, 0.0, 0.0, 0.0],
                [1.5181065386568227e8, 1.5181065386568227e8, 1.534561321107214e8, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.526333929882019e8, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.526333929882019e8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.526333929882019e8],
            ]
        )
        np.testing.assert_allclose(stiffness, expected, rtol=2.0e-10, atol=1.0e-4)

    def test_solid_junction_and_kj_round_trip(self):
        path = self.examples / "two_connection_cube_junction.sc"
        model = read_solid_junction(path)
        stiffness = calculate_junction_stiffness(model)
        rigid = rigid_connection_modes(stiffness.connection_points)
        relative_rigid_residual = (
            np.linalg.norm(stiffness.matrix @ rigid)
            / np.linalg.norm(stiffness.matrix)
        )
        self.assertEqual(stiffness.matrix.shape, (12, 12))
        self.assertLess(relative_rigid_residual, 1.0e-12)
        self.assertEqual(np.linalg.matrix_rank(stiffness.matrix, tol=1.0e-7), 6)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "junction.kj"
            write_junction_stiffness(output, stiffness)
            loaded = read_junction_stiffness(output)
        np.testing.assert_allclose(loaded.matrix, stiffness.matrix, rtol=0.0, atol=0.0)

    def test_tet4_solid_junction_mesh_is_supported(self):
        solid_input = """0 0 0 0

3 4 1 1 0 0

1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0

1 1 1 2 3 4

1 0 1
0.0 1.0
1000.0 0.25

0.1666666666666667
"""
        connection_input = """1 1

1 1 0.0 0.3333333333333333 0.3333333333333333 1 0 0 0 1 0 0 0 1
1 3 4
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tet4_junction.sc"
            path.write_text(solid_input)
            Path(str(path) + ".msg").write_text(connection_input)
            model = read_solid_junction(path)
            stiffness = assemble_solid_stiffness(model).toarray()
        self.assertEqual(tuple(len(element) for element in model.elements), (4,))
        self.assertEqual(model.interface_faces[0].shape, (1, 3))
        self.assertEqual(stiffness.shape, (12, 12))
        eigenvalues = np.linalg.eigvalsh(stiffness)
        self.assertTrue(np.all(eigenvalues[6:] > 1.0e-8))
        np.testing.assert_allclose(eigenvalues[:6], 0.0, atol=1.0e-10)

        cube = read_solid_junction(
            self.examples / "linear_two_connection_cube_junction.sc"
        )
        junction = calculate_junction_stiffness(cube)
        self.assertEqual(junction.matrix.shape, (12, 12))
        self.assertEqual(np.linalg.matrix_rank(junction.matrix, tol=1.0e-7), 6)

    def test_mode_one_and_mode_two_use_the_same_hybrid_path(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            names = (
                "hybrid_cube_mode1.sc",
                "hybrid_cube_mode1.sc.msg",
                "hybrid_cube_mode2.sc",
                "hybrid_cube_mode2.sc.msg",
                "two_connection_cube_junction.sc",
                "two_connection_cube_junction.sc.msg",
            )
            for name in names:
                shutil.copy2(self.examples / name, work / name)
            _, _, mode_one = homogenize(work / "hybrid_cube_mode1.sc")
            self.assertTrue((work / "two_connection_cube_junction.sc.kj").exists())
            _, _, mode_two = homogenize(work / "hybrid_cube_mode2.sc")
        np.testing.assert_allclose(
            mode_one.effective_stiffness,
            mode_two.effective_stiffness,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_bcch_euler_is_slightly_stiffer_than_timoshenko(self):
        directory = self.examples / "bcch_lambda06"
        results = {}
        for theory in ("euler", "timoshenko"):
            for flag in (0, 2):
                _, _, result = homogenize(
                    directory / f"bcch_{theory}_flag{flag}.sc"
                )
                results[(theory, flag)] = result.engineering_constants["E1"]
        np.testing.assert_allclose(
            [results[("euler", 0)], results[("euler", 2)]],
            [53.830217703953, 72.653749548306],
            rtol=2.0e-10,
        )
        self.assertGreater(results[("euler", 0)], results[("timoshenko", 0)])
        self.assertGreater(results[("euler", 2)], results[("timoshenko", 2)])

    def test_solver_uses_derivation_sign_for_v_hat_0(self):
        e = np.diag(np.arange(1.0, 7.0))
        d_h_epsilon = np.diag(np.linspace(0.2, 1.2, 6))
        d_epsilon_epsilon = 5.0 * np.eye(6)
        terms = HomogenizationTerms(
            e=e,
            d_h_epsilon=d_h_epsilon,
            d_epsilon_epsilon=d_epsilon_epsilon,
            d_h_lambda=np.zeros((6, 6)),
            f_h_lambda=np.zeros((6, 6)),
        )
        stiffness, v_hat_0, v_0 = solve_homogenization(
            terms, np.eye(6), volume=2.0
        )
        expected_v_0 = -np.linalg.solve(e, d_h_epsilon)
        expected_stiffness = (
            d_epsilon_epsilon
            + 2.0 * expected_v_0.T @ d_h_epsilon
            + expected_v_0.T @ e @ expected_v_0
        ) / 2.0
        np.testing.assert_allclose(v_hat_0, expected_v_0)
        np.testing.assert_allclose(v_0, expected_v_0)
        np.testing.assert_allclose(stiffness, expected_stiffness)

    def test_sparse_solver_ignores_zero_euler_constraint_rows(self):
        number_of_dofs = 2001
        constraint = np.zeros((6, number_of_dofs))
        constraint[:3, :3] = np.eye(3)
        terms = HomogenizationTerms(
            e=sparse.eye(number_of_dofs, format="csr"),
            d_h_epsilon=np.zeros((number_of_dofs, 6)),
            d_epsilon_epsilon=5.0 * np.eye(6),
            d_h_lambda=constraint,
            f_h_lambda=np.zeros((6, 6)),
        )
        stiffness, v_hat_0, v_0 = solve_homogenization(
            terms, sparse.eye(number_of_dofs, format="csr"), volume=2.0
        )
        np.testing.assert_allclose(stiffness, 2.5 * np.eye(6))
        np.testing.assert_allclose(v_hat_0, 0.0)
        np.testing.assert_allclose(v_0, 0.0)

    def test_historical_fluctuation_alias_has_old_sign(self):
        _, _, result = homogenize(
            self.examples / "rigid_timoshenko_cross.sc"
        )
        np.testing.assert_allclose(result.full_fluctuation, -result.v_0)
        np.testing.assert_allclose(result.reduced_fluctuation, -result.v_hat_0)

    def test_boundary_connection_uses_unwrapped_position(self):
        nodes = np.array([[-0.5, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        stiffness = JunctionStiffness(
            connection_points=(
                JunctionConnectionPoint(1, np.zeros(3), np.eye(3)),
            ),
            matrix=np.zeros((6, 6)),
        )
        instance = JunctionInstance(1, 1, np.array([0.5, 0.0, 0.0]), np.eye(3))
        connection = JunctionConnection(1, 1, 1, 1, np.array([1.0, 0.0, 0.0]))
        b_v, b_epsilon = connection_matrices(
            number_of_dofs=12,
            nodes=nodes,
            elements={1: np.array([0, 1])},
            instance=instance,
            connections=[connection],
            stiffness=stiffness,
        )
        np.testing.assert_allclose(b_v[:, :6], np.eye(6))
        np.testing.assert_allclose(
            b_epsilon[:3],
            np.array(
                [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                 [0.0, 0.0, 0.0, 0.0, 0.25, 0.0]]
            ),
        )
        np.testing.assert_allclose(b_epsilon[3:], 0.0)

    def _ownership_input(self, beam_positions=(), junction_positions=()):
        nodes = [
            np.array([-0.5, -0.5, -0.5]),
            np.array([0.5, 0.5, 0.5]),
        ]
        connectivity = []
        for position in beam_positions:
            start = np.asarray(position, dtype=float)
            connectivity.append(np.array([len(nodes), len(nodes) + 1]))
            nodes.extend((start, start + np.array([0.25, 0.0, 0.0])))
        element_ids = np.arange(1, len(connectivity) + 1, dtype=np.int64)
        model = StructuralGenomeInput(
            path=Path("ownership.sc"), analysis=0, element_flag=2,
            transformation_flag=1, temperature_flag=0,
            junction_flag=2 if junction_positions else 0, dimension=3,
            nodes=np.asarray(nodes), element_ids=element_ids,
            material_ids=np.ones(len(connectivity), dtype=np.int64),
            connectivity=tuple(connectivity), orientations={}, periodic_pairs=[],
            number_of_materials=1, tail_records=(), volume=1.0,
        )
        beam_type = BeamType(
            1, BeamTheory.EULER_BERNOULLI, 3, np.eye(4)
        )
        instances = {
            identifier: JunctionInstance(
                identifier, 1, np.asarray(position, dtype=float), np.eye(3)
            )
            for identifier, position in enumerate(junction_positions, start=1)
        }
        supplement = HybridSupplement(
            version=1, beam_types={1: beam_type},
            beam_assignments={int(identifier): 1 for identifier in element_ids},
            junction_types=(
                {1: JunctionType(1, 1, Path("junction.kj"))}
                if junction_positions else {}
            ),
            junction_instances=instances, junction_connections=(),
            beam_recovery={},
        )
        return model, supplement

    def test_automatic_periodic_beam_ownership_on_face_and_edge(self):
        cases = (
            (([0.0, -0.5, 0.0], [0.0, 0.5, 0.0]), 2),
            ((
                [0.0, -0.5, -0.5], [0.0, -0.5, 0.5],
                [0.0, 0.5, -0.5], [0.0, 0.5, 0.5],
            ), 4),
        )
        for positions, input_count in cases:
            with self.subTest(input_count=input_count):
                model, supplement = self._ownership_input(positions)
                owned_model, owned_supplement, audit = apply_periodic_ownership(
                    model, supplement
                )
                self.assertEqual(audit.input_beams, input_count)
                self.assertEqual(audit.owned_beams, 1)
                self.assertEqual(len(build_beam_discretization(
                    owned_model, owned_supplement
                )[1]), 1)

    def test_automatic_periodic_junction_ownership_on_face_edge_and_corner(self):
        for axes in (1, 2, 3):
            positions = []
            for image in np.ndindex(*(2,) * axes):
                point = np.zeros(3)
                point[:axes] = [(-0.5, 0.5)[value] for value in image]
                positions.append(point)
            model, supplement = self._ownership_input(
                beam_positions=(), junction_positions=positions
            )
            _, owned, audit = apply_periodic_ownership(model, supplement)
            self.assertEqual(audit.input_junctions, 2 ** axes)
            self.assertEqual(audit.owned_junctions, 1)
            self.assertEqual(len(owned.junction_instances), 1)
            np.testing.assert_allclose(
                next(iter(owned.junction_instances.values())).origin[:axes],
                -0.5,
            )

    def test_boundary_beam_and_junction_choose_same_representative(self):
        positions = ([0.0, -0.5, 0.0], [0.0, 0.5, 0.0])
        model, supplement = self._ownership_input(positions, positions)
        supplement = HybridSupplement(
            **{
                **supplement.__dict__,
                "junction_connections": tuple(
                    JunctionConnection(index, 1, index, 1, np.zeros(3))
                    for index in (1, 2)
                ),
            }
        )
        owned_model, owned_supplement, audit = apply_periodic_ownership(
            model, supplement
        )
        self.assertEqual((audit.owned_beams, audit.owned_junctions), (1, 1))
        connection = owned_supplement.junction_connections[0]
        instance = owned_supplement.junction_instances[connection.junction_id]
        endpoint = owned_model.nodes[
            owned_model.connectivity[0][connection.endpoint - 1]
        ]
        np.testing.assert_allclose(endpoint, instance.origin)
        np.testing.assert_allclose(connection.image_shift, 0.0)

    def test_periodic_connection_shift_is_inferred_without_opposite_node(self):
        beam_position = [0.0, 0.5, 0.0]
        junction_position = [0.0, -0.5, 0.0]
        model, supplement = self._ownership_input(
            [beam_position], [junction_position]
        )
        supplement = HybridSupplement(
            **{
                **supplement.__dict__,
                "junction_connections": (
                    JunctionConnection(1, 1, 1, 1, np.zeros(3)),
                ),
            }
        )
        stiffness = JunctionStiffness(
            connection_points=(
                JunctionConnectionPoint(1, np.zeros(3), np.eye(3)),
            ),
            matrix=np.zeros((6, 6)),
        )
        completed = complete_periodic_connection_shifts(
            model, supplement, {1: stiffness}
        )
        np.testing.assert_allclose(
            completed.junction_connections[0].image_shift,
            [0.0, -1.0, 0.0],
        )


    def test_four_value_control_record_defaults_to_rigid_mode(self):
        source = (self.examples / "rigid_timoshenko_cross.sc").read_text()
        source = source.replace("0 2 1 0 0", "0 2 1 0", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sc"
            path.write_text(source)
            model = read_structural_genome(path)
        self.assertEqual(model.junction_flag, 0)

    def test_beam_input_accepts_only_physical_endpoints(self):
        source = (self.examples / "rigid_timoshenko_cross.sc").read_text()
        source = source.replace("1 1 1 2\n", "1 1 1 2 7\n", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "internal_node.sc"
            path.write_text(source)
            with self.assertRaisesRegex(
                ValueError, "requires exactly two endpoint nodes"
            ):
                read_structural_genome(path)

    def test_rigid_flag_rejects_hybrid_junction_records(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for name in (
                "hybrid_cube_mode2.sc",
                "hybrid_cube_mode2.sc.msg",
                "two_connection_cube_junction.sc.kj",
            ):
                shutil.copy2(self.examples / name, work / name)
            path = work / "hybrid_cube_mode2.sc"
            path.write_text(path.read_text().replace("0 2 1 0 2", "0 2 1 0 0", 1))
            with self.assertRaisesRegex(ValueError, "selects a pure-beam model"):
                homogenize(path)

    def test_periodic_translations_must_span_three_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            path = work / "incomplete_periodicity.sc"
            text = (self.examples / "rigid_timoshenko_cross.sc").read_text()
            text = text.replace("3 7 6 1 3 0", "3 7 6 1 2 0", 1)
            text = text.replace("7 6\n", "", 1)
            path.write_text(text)
            shutil.copy2(
                self.examples / "rigid_timoshenko_cross.sc.msg",
                Path(str(path) + ".msg"),
            )
            with self.assertRaisesRegex(ValueError, "span only 2 independent directions"):
                homogenize(path)

    def test_junction_connection_reports_invalid_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for name in (
                "hybrid_cube_mode2.sc",
                "hybrid_cube_mode2.sc.msg",
                "two_connection_cube_junction.sc.kj",
            ):
                shutil.copy2(self.examples / name, work / name)
            supplement = work / "hybrid_cube_mode2.sc.msg"
            text = supplement.read_text().replace(
                "1 1 1 1 0 0 0", "1 3 1 1 0 0 0", 1
            )
            supplement.write_text(text)
            with self.assertRaisesRegex(ValueError, "missing=.*invalid"):
                homogenize(work / "hybrid_cube_mode2.sc")

    def test_solid_junction_reports_off_plane_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for name in (
                "two_connection_cube_junction.sc",
                "two_connection_cube_junction.sc.msg",
            ):
                shutil.copy2(self.examples / name, work / name)
            supplement = work / "two_connection_cube_junction.sc.msg"
            text = supplement.read_text().replace(
                "1 2 0.0 0.5 0.5", "1 2 0.25 0.5 0.5", 1
            )
            supplement.write_text(text)
            with self.assertRaisesRegex(ValueError, "interface does not lie in its plane"):
                read_solid_junction(work / "two_connection_cube_junction.sc")


if __name__ == "__main__":
    unittest.main()
