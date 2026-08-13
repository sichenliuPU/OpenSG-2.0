from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

EXAMPLES_ROOT = Path(__file__).parents[2] / "examples"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.append(str(EXAMPLES_ROOT))

from fe_jax.beam import beam_frame
from fe_jax.junction import JunctionConnectionPoint
from fe_jax.junction_solid import analyze_junction
from fe_jax.sc_hybrid_input import read_solid_junction
from tools.junction_boolean import (
    build_boolean_junction,
    write_solid_junction_input,
)


class TestJunctionInputBuilder(unittest.TestCase):
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
            self.assertTrue(
                all(len(element) == number_of_nodes for element in model.elements)
            )
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


if __name__ == "__main__":
    unittest.main()
