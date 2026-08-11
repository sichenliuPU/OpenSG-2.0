# Beam and hybrid structure-gene input

The beam and hybrid solver uses the standard OpenSG command for all junction modes:

```text
opensg model.sc 3D H
```

The first input record accepts an optional fifth control value:

```text
analysis elem_flag trans_flag temp_flag junction_flag
```

`junction_flag` has the following values:

```text
0  pure beam model with rigid shared-node junctions
1  hybrid model; calculate junction stiffness from referenced 3D solid inputs
2  hybrid model; read referenced junction-stiffness files
```

If the fifth value is omitted, it defaults to zero. Beam homogenization requires
`elem_flag=2`. The standard node, element, orientation, periodic-pair, material,
and volume sections retain their existing order.

Every beam element record contains only its two physical endpoints:

```text
element_id material_id endpoint_1 endpoint_2
```

OpenSG creates the interpolation nodes internally. Periodic pairs and junction
connections therefore always reference user-supplied endpoint nodes.

## Beam and junction supplement

OpenSG reads `model.sc.msg` automatically. Its header is:

```text
version nbeamtype nbeamassign njunctiontype njunction nconnection jtrans_flag
```

Each beam type starts with:

```text
beam_type_id theory interpolation_node_count nstrain
```

The interpolation-node count is 3 for Euler–Bernoulli and 4 for Timoshenko;
these nodes are generated rather than listed in the main connectivity. The
record is followed by `nstrain` full rows of its symmetric section-stiffness matrix.
The theory values are:

```text
0  three-node Hermite Euler--Bernoulli generated from two endpoints
1  four-node Lagrange Timoshenko generated from two endpoints
```

Timoshenko section components are ordered as extension, shear 12, shear 13,
twist, bend 2, and bend 3. No alpha parameter is read or used.

Beam assignments follow the beam types:

```text
element_id beam_type_id
```

For junction modes 1 and 2, junction types follow the assignments:

```text
junction_type_id nconnectionpoint source_file
```

Mode 1 source files are standard 3D solid `.sc` files. Mode 2 source files are
`.kj` files.
Junction instances are:

```text
junction_id junction_type_id x y z
```

When `jtrans_flag=1`, one standard three-point orientation record follows for
each junction. Connections are then listed as:

```text
junction_id connection_point_id element_id endpoint shift_x shift_y shift_z
```

Boundary junctions use the same records as internal junctions. The shift is
zero when the beam endpoint already coincides with its connection point.
Otherwise, enter the periodic translation that makes the endpoint coincide
with the connection point.
OpenSG then applies the beam periodic mapping; the local 3D solid-junction mesh
does not require periodic constraints.

## Solid junction connection points and interfaces

A mode-1 junction source `junction.sc` uses standard OpenSG 3D solid
connectivity and isotropic material records. TET4 and TET10 elements are
supported. Its companion `junction.sc.msg` begins with:

```text
version nconnectionpoint
```

Each connection point begins with:

```text
connection_point_id nface x y z b11 b12 b13 b21 b22 b23 b31 b32 b33
```

For TET4 elements, it is followed by `nface` TRI3 interface records:

```text
node_1 node_2 node_3
```

For TET10 elements, use TRI6 interface records:

```text
node_1 node_2 node_3 node_12 node_23 node_31
```

Interfaces may meet along shared perimeter nodes. OpenSG uses surface
integration consistent with the TRI3 or TRI6 interface interpolation. It writes the
calculated stiffness to `junction.sc.kj` and immediately uses it in the same
homogenization run.

## Output

All three modes write:

```text
model.sc.ech
model.sc.k
```

Mode 1 additionally writes one reusable `.kj` file for every unique solid
junction type.

## Input validation

Validation is performed before homogenization. OpenSG reports an input error
and identifies the offending flag, element, node, junction, connection, or file when
it detects an invalid model. Checks include:

- supported and mutually consistent analysis, element, temperature, and
  junction flags;
- complete beam assignments, supported connectivity, orientations, finite
  coordinates, positive-definite section stiffness, and unused nodes;
- valid periodic pairs, opposite-face geometry, matching beam connectivity,
  and three independent periodic translations assembled from node pairs and
  junction image shifts;
- agreement among junction types, instances, connections, beam endpoints,
  source files, and periodic-image connection-point positions;
- valid `.kj` dimensions, symmetry, positive deformational modes, rank, and
  rigid modes;
- solid material identifiers, positive elastic properties, TET4/TET10 element
  Jacobians, exterior TRI3/TRI6 interfaces, planar interface geometry, and
  valid connection-point frames.

A computed zero-energy mechanism is written to the `.k` file and produces a
command-line warning because a mechanism can be intentional, although it often
indicates missing connectivity or periodic constraints.
