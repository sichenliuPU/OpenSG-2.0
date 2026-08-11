"""Optional preprocessing utility for solid-junction input generation.

The homogenization and localization solvers do not import this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .beam import FloatArray, IntArray
from .junction import JunctionConnectionPoint
from .junction_solid import SolidJunctionModel


def _triangle_area(coordinates: FloatArray) -> float:
    if len(coordinates) == 3:
        return 0.5 * float(np.linalg.norm(np.cross(
            coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]
        )))
    # The Boolean TET10 meshes use straight edges, so the corner triangle has
    # the same area as its TRI6 representation.
    return _triangle_area(coordinates[:3])


def build_boolean_junction(
    connection_points: tuple[JunctionConnectionPoint, ...],
    cross_section: str,
    section_size: float,
    mesh_size: float,
    element_flag: int,
    material_id: int,
    material: tuple[float, float],
    interface_tolerance: float = 1.0e-7,
) -> SolidJunctionModel:
    """Build a Boolean union of oriented beam stubs and mesh it with tetrahedra.

    ``element_flag=0`` generates TET4/TRI3 and ``element_flag=1`` generates
    TET10/TRI6. ``section_size`` is square half-width or circular radius.
    """

    import gmsh

    if len(connection_points) < 2:
        raise ValueError("A Boolean junction requires at least two connections.")
    if cross_section not in {"square", "circular"}:
        raise ValueError("Boolean junction sections must be square or circular.")
    if section_size <= 0.0 or mesh_size <= 0.0:
        raise ValueError("Boolean junction section and mesh sizes must be positive.")
    if element_flag not in (0, 1):
        raise ValueError("Junction element_flag must be 0 (TET4) or 1 (TET10).")

    lengths = []
    for point in connection_points:
        direction = point.frame[0]
        length = float(np.dot(point.origin, direction))
        transverse = point.origin - length * direction
        tolerance = 1.0e-8 * max(abs(length), section_size, 1.0)
        if length <= section_size or np.linalg.norm(transverse) > tolerance:
            raise ValueError(
                f"Connection point {point.identifier} is not on its positive local axis."
            )
        lengths.append(length)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 0)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1.0e-8)
        gmsh.model.add(f"junction_{len(connection_points)}_connections")
        volumes: list[tuple[int, int]] = []
        back = section_size
        for point, length in zip(connection_points, lengths, strict=True):
            direction = point.frame[0]
            if cross_section == "circular":
                start = -back * direction
                axis = (length + back) * direction
                tag = gmsh.model.occ.addCylinder(
                    *start.tolist(), *axis.tolist(), section_size
                )
            else:
                tag = gmsh.model.occ.addBox(
                    -back, -section_size, -section_size,
                    length + back, 2.0 * section_size, 2.0 * section_size,
                )
                rotation_vector = Rotation.from_matrix(point.frame.T).as_rotvec()
                angle = float(np.linalg.norm(rotation_vector))
                if angle > 1.0e-14:
                    axis = rotation_vector / angle
                    gmsh.model.occ.rotate(
                        [(3, tag)], 0.0, 0.0, 0.0, *axis.tolist(), angle
                    )
            volumes.append((3, tag))

        try:
            union = [volumes[0]]
            for tool in volumes[1:]:
                union, _ = gmsh.model.occ.fuse(
                    union, [tool], removeObject=True, removeTool=True
                )
                gmsh.model.occ.synchronize()
        except Exception:
            gmsh.model.occ.synchronize()
            union, _ = gmsh.model.occ.fragment(volumes, [])
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        volume_tags = sorted({tag for dim, tag in union if dim == 3})
        if not volume_tags:
            volume_tags = [tag for _dim, tag in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3, volume_tags, 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber(
            "Mesh.SecondOrderLinear", 1 if cross_section == "circular" else 0
        )
        gmsh.model.mesh.generate(3)
        if element_flag == 1:
            gmsh.model.mesh.setOrder(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        order = np.argsort(node_tags)
        tags = np.asarray(node_tags, dtype=np.int64)[order]
        xyz = xyz[order]
        lookup = {int(tag): index for index, tag in enumerate(tags)}

        def elements_of_type(kind: int, dimension: int, width: int) -> IntArray:
            types, _, blocks = gmsh.model.mesh.getElements(dimension)
            selected = []
            for element_type, block in zip(types, blocks, strict=True):
                if int(element_type) == kind:
                    raw = np.asarray(block, dtype=np.int64).reshape(-1, width)
                    selected.append(np.asarray(
                        [[lookup[int(value)] for value in row] for row in raw],
                        dtype=np.int64,
                    ))
            return (
                np.vstack(selected)
                if selected else np.empty((0, width), dtype=np.int64)
            )

        if element_flag == 0:
            elements = elements_of_type(4, 3, 4)
            surface_faces = elements_of_type(2, 2, 3)
        else:
            elements = elements_of_type(11, 3, 10)
            # Gmsh: 1,2,3,4,12,23,31,14,34,24.
            # OpenSG: 1,2,3,4,12,23,31,14,24,34.
            elements = elements[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
            surface_faces = elements_of_type(9, 2, 6)
        if len(elements) == 0 or len(surface_faces) == 0:
            raise RuntimeError("Gmsh did not generate the requested junction elements.")

        groups: list[list[IntArray]] = [[] for _ in connection_points]
        ownership = np.zeros(len(surface_faces), dtype=np.int64)
        for face_index, face in enumerate(surface_faces):
            points = xyz[face]
            for connection_index, (point, length) in enumerate(
                zip(connection_points, lengths, strict=True)
            ):
                local = points @ point.frame.T
                if cross_section == "circular":
                    in_section = np.all(
                        np.linalg.norm(local[:, 1:3], axis=1)
                        <= section_size + interface_tolerance
                    )
                else:
                    in_section = np.all(
                        np.abs(local[:, 1:]) <= section_size + interface_tolerance
                    )
                if (
                    np.all(np.abs(local[:, 0] - length) <= interface_tolerance)
                    and in_section
                ):
                    groups[connection_index].append(face)
                    ownership[face_index] += 1
        if np.any(ownership > 1):
            raise RuntimeError("A junction interface face has multiple owners.")
        width = 3 if element_flag == 0 else 6
        interfaces = tuple(
            np.asarray(group, dtype=np.int64).reshape(-1, width) for group in groups
        )
        if any(len(group) == 0 for group in interfaces):
            raise RuntimeError(
                "At least one junction connection has no interface faces: "
                f"{[len(group) for group in interfaces]}."
            )
        expected_area = (
            np.pi * section_size**2
            if cross_section == "circular" else (2.0 * section_size) ** 2
        )
        areas = np.asarray([
            sum(_triangle_area(xyz[face]) for face in group)
            for group in interfaces
        ])
        relative_tolerance = 6.0e-2 if cross_section == "circular" else 8.0e-3
        if not np.allclose(
            areas, expected_area, rtol=relative_tolerance, atol=1.0e-12
        ):
            raise RuntimeError(f"Invalid junction interface areas: {areas.tolist()}.")

        return SolidJunctionModel(
            nodes=xyz,
            elements=tuple(np.asarray(row, dtype=np.int64) for row in elements),
            material_ids=np.full(len(elements), material_id, dtype=np.int64),
            materials={material_id: material},
            connection_points=connection_points,
            interface_faces=interfaces,
        )
    finally:
        gmsh.finalize()


def write_solid_junction_input(
    path: str | Path, model: SolidJunctionModel
) -> tuple[Path, Path]:
    """Write a generated TET4/TET10 junction as reusable OpenSG input files."""

    path = Path(path)
    material_ids = sorted(model.materials)
    if material_ids != list(range(1, len(material_ids) + 1)):
        raise ValueError("Solid junction material IDs must be consecutive from one.")
    element_sizes = {len(element) for element in model.elements}
    if not element_sizes <= {4, 10}:
        raise ValueError("Solid junction input supports only TET4 and TET10 elements.")

    lines = [
        "0 0 0 0",
        "",
        f"3 {len(model.nodes)} {len(model.elements)} {len(material_ids)} 0 0",
        "",
    ]
    lines.extend(
        f"{identifier} " + " ".join(f"{value:.16e}" for value in coordinates)
        for identifier, coordinates in enumerate(model.nodes, start=1)
    )
    lines.append("")
    for identifier, (element, material_id) in enumerate(
        zip(model.elements, model.material_ids, strict=True), start=1
    ):
        nodes = (np.asarray(element, dtype=np.int64) + 1).tolist()
        if len(nodes) == 10:
            nodes = nodes[:4] + [0] + nodes[4:]
        lines.append(
            f"{identifier} {int(material_id)} "
            + " ".join(str(node) for node in nodes)
        )
    lines.append("")
    for material_id in material_ids:
        young, poisson = model.materials[material_id]
        lines.extend((
            f"{material_id} 0 1",
            "0.0 1.0",
            f"{young:.16e} {poisson:.16e}",
            "",
        ))
    volume = sum(
        abs(float(np.linalg.det(
            (model.nodes[np.asarray(element)[:4]][1:]
             - model.nodes[np.asarray(element)[0]]).T
        ))) / 6.0
        for element in model.elements
    )
    lines.append(f"{volume:.16e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    interface_path = Path(str(path) + ".msg")
    interface_lines = [f"1 {len(model.connection_points)}", ""]
    for point, faces in zip(
        model.connection_points, model.interface_faces, strict=True
    ):
        geometry = np.concatenate((point.origin, point.frame.reshape(-1)))
        interface_lines.append(
            f"{point.identifier} {len(faces)} "
            + " ".join(f"{value:.16e}" for value in geometry)
        )
        interface_lines.extend(
            " ".join(str(int(node) + 1) for node in face) for face in faces
        )
        interface_lines.append("")
    interface_path.write_text(
        "\n".join(interface_lines) + "\n", encoding="utf-8"
    )
    return path, interface_path
