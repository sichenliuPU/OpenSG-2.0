"""OpenSG-style output for beam and hybrid homogenization."""

from __future__ import annotations

from pathlib import Path

from .hybrid_homogenization import HomogenizationResult
from .sc_hybrid_input import HybridSupplement, StructuralGenomeInput


def _matrix_lines(matrix) -> list[str]:
    return [" ".join(f"{value:16.7E}" for value in row) for row in matrix]


def write_effective_properties(
    path: str | Path,
    model: StructuralGenomeInput,
    result: HomogenizationResult,
) -> None:
    """Write stiffness, compliance, properties, and timing to ``.sc.k``."""

    constants = result.engineering_constants
    lines = [
        " The Effective Stiffness Matrix",
        " --------------------------------------------",
        *_matrix_lines(result.effective_stiffness),
        "",
        " The Effective Compliance Matrix",
        " --------------------------------------------",
        *_matrix_lines(result.effective_compliance),
        "",
        " The Engineering Constants (Approximated as Orthotropic)",
        " ----------------------------------------------------------",
        f"  E1  = {constants['E1']:16.7E}",
        f"  E2  = {constants['E2']:16.7E}",
        f"  E3  = {constants['E3']:16.7E}",
        f"  G12 = {constants['G12']:16.7E}",
        f"  G13 = {constants['G13']:16.7E}",
        f"  G23 = {constants['G23']:16.7E}",
        f"  nu12= {constants['nu12']:16.7E}",
        f"  nu13= {constants['nu13']:16.7E}",
        f"  nu23= {constants['nu23']:16.7E}",
        "",
        f" Junction mode             = {model.junction_flag}",
        f" Input beam endpoint nodes = {len(model.nodes)}",
        f" Internal analysis nodes   = {result.number_of_full_dofs // 6}",
        f" Full beam DOFs            = {result.number_of_full_dofs}",
        f" Independent beam DOFs     = {result.number_of_independent_dofs}",
        f" Input/owned beams         = {result.periodic_ownership.input_beams}/"
        f"{result.periodic_ownership.owned_beams}",
        f" Input/owned junctions     = {result.periodic_ownership.input_junctions}/"
        f"{result.periodic_ownership.owned_junctions}",
        f" Junction instances        = {result.number_of_junctions}",
        f" Zero-energy mechanism     = {str(result.has_mechanism).lower()}",
        f" Junction analysis time [s]= {result.junction_analysis_time:.7E}",
        f" Homogenization time [s]   = {result.homogenization_time:.7E}",
        f" Total time [s]            = {result.total_time:.7E}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_echo(
    path: str | Path,
    model: StructuralGenomeInput,
    supplement: HybridSupplement,
) -> None:
    """Write a concise echo of the interpreted hybrid input."""

    lines = [
        " Problem control parameters: analysis elem_type trans_flag temp_flag junction_flag",
        " --------------------------------------------",
        f" {model.analysis:10d}{model.element_flag:10d}{model.transformation_flag:10d}"
        f"{model.temperature_flag:10d}{model.junction_flag:10d}",
        "",
        " Mesh control summary",
        " --------------------------------------------",
        f" dimension              = {model.dimension}",
        f" beam endpoint nodes    = {len(model.nodes)}",
        f" elements               = {len(model.element_ids)}",
        f" periodic slave nodes   = {len(model.periodic_pairs)}",
        f" beam types             = {len(supplement.beam_types)}",
        f" junction types         = {len(supplement.junction_types)}",
        f" junction instances     = {len(supplement.junction_instances)}",
        f" junction connections   = {len(supplement.junction_connections)}",
        f" structure-gene volume    = {model.volume:.16E}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
