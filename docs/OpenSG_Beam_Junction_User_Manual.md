# OpenSG Beam and Beam–Junction Homogenization User Manual

Version 0.2
Applicable analysis: linear-elastic homogenization and dehomogenization of
three-dimensional periodic beam-based structure genes.

## 1. Purpose and scope

OpenSG calculates the effective elastic properties of a structure gene (SG). A model is described by input files. 
The program supports:

- three-node Hermite Euler–Bernoulli beams generated from two supplied endpoints;
- four-node Lagrange Timoshenko beams generated from two supplied endpoints;
- rigid beam junctions;
- junction stiffness calculated from linear or quadratic tetrahedral solid models;
- previously calculated, reusable junction-stiffness files;
- internal junctions and periodically owned face, edge, and corner junctions;
- full (6\times6) effective stiffness and compliance matrices;
- SwiftComp-compatible global-field input for dehomogenization;
- Euler--Bernoulli and Timoshenko beam recovery through VABS; and
- three-dimensional junction displacement, strain, and stress recovery.

The current solid-junction analysis uses the standard OpenSG `.sc` mesh input.
It supports linear TET4 elements with TRI3 interfaces and quadratic TET10
elements with TRI6 interfaces. All quantities must use one consistent unit
system. OpenSG does not assign units.

## 2. Files used by an analysis

Every beam analysis requires two files with the same base name:

```text
model.sc
model.sc.msg
```

The `.sc` file contains the SG mesh, element connectivity, orientations,
periodic node pairs, material records, and SG volume. The `.sc.msg` file
contains the beam section properties and, when applicable, the junction
definitions.

Depending on the selected junction mode, additional files are required:

| Junction flag | Required additional files |
|---:|---|
| 0 | None |
| 1 | One 3D solid `.sc` file and companion `.sc.msg` file for each junction type |
| 2 | One `.kj` junction-stiffness file for each junction type |

Dehomogenization additionally requires:

| File | Purpose |
|---|---|
| `model.sc.glb` | Macroscopic displacement, deformation, and strain or stress |
| VABS section source | Three-dimensional recovery for every beam type |
| Same-basename junction `.sc/.sc.msg` | Junction recovery when flag 2 reads a `.kj` file |

For example, a flag-2 source named `junction.sc.kj` requires `junction.sc` and
`junction.sc.msg` for dehomogenization. Homogenization itself requires only
the `.kj` file.

Relative junction filenames are interpreted relative to the directory that
contains `model.sc.msg`.

Blank lines are allowed. Text following `#` or `!` is treated as a comment.

## 3. Installation and command

Open a terminal in the OpenSG source directory and install the package into the
active Python environment:

```text
python -m pip install -e .
```

Python 3.11 or later is required. The environment must contain the dependencies
listed in `pyproject.toml`.

The homogenization and dehomogenization solvers do not require Gmsh or
Matplotlib when users supply their own meshes. Install optional preprocessing
or example dependencies only when needed:

```text
python -m pip install -e ".[input-builders]"
python -m pip install -e ".[examples]"
```

Run a three-dimensional homogenization with:

```text
opensg model.sc 3D H
```

The command may be issued from any directory if the path to `model.sc` is
given. For example:

```text
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag2.sc 3D H
```

The program writes:

```text
model.sc.ech
model.sc.k
```

The `.ech` file summarizes how the input was interpreted. The `.k` file
contains the effective stiffness matrix, compliance matrix, engineering
constants, model size, junction mode, mechanism status, and timing.

Junction flag 1 also writes a reusable `.kj` file beside every referenced
solid-junction `.sc` file.

Run dehomogenization with:

```text
opensg model.sc 3D L
```

This command is standalone: it reconstructs the required operators from the
supplied inputs and does not require an earlier `3D H` run or a `.k` file.
Use `--stations N` to select the number of axial VABS stations per beam and
`--vabs PATH` to select the VABS executable. The command writes:

```text
model.sc.u
model.sc.sn
```

## 4. Choosing the beam and junction models

The beam theory is selected in `model.sc.msg`. The junction treatment is
selected in the first record of `model.sc`.

### 4.1 Beam theory identifiers

| Beam theory | Identifier | Nodes |
|---|---:|---:|
| Euler–Bernoulli | 0 | 3 |
| Timoshenko | 1 | 4 |

The user supplies only the two physical endpoints of every beam element in
`model.sc`. OpenSG places the additional interpolation nodes automatically.

### 4.2 Junction flags

| Flag | Definition |
|---:|---|
| 0 | Pure beam model. Beams meeting at one junction share one node and are rigidly connected. |
| 1 | Hybrid model. Junction stiffness is calculated from 3D solid-junction inputs. |
| 2 | Hybrid model. Junction stiffness is read from referenced `.kj` files. |

Flags 1 and 2 use the same beam SG and junction connections. They differ only
in the source used for each junction type. A flag-1 run creates the files that
can subsequently be used by flag 2.

## 5. Main input file: `model.sc`

Sections must appear in the order described below.

### 5.1 Analysis-control record

```text
analysis elem_flag trans_flag temp_flag junction_flag
```

For the present beam homogenization capability:

```text
0 2 trans_flag 0 junction_flag
```

The values are:

| Position | Name | Permitted value |
|---:|---|---|
| 1 | `analysis` | 0: homogenization |
| 2 | `elem_flag` | 2: beam elements |
| 3 | `trans_flag` | 0: automatic orientations; 1: orientation records supplied |
| 4 | `temp_flag` | 0: no temperature dependence |
| 5 | `junction_flag` | 0, 1, or 2 as defined above |

Examples:

```text
0 2 1 0 0
0 2 1 0 1
0 2 1 0 2
```

The fifth value may be omitted only for a flag-0 model; omission defaults to
zero.

### 5.2 Mesh-control record

```text
dimension nnode nelem nmaterial nslave nlayer
```

For example:

```text
3 66 44 1 0 0
```

This record declares a three-dimensional SG with 66 endpoint nodes, 44 elements, one
material, no explicit slave nodes, and no layer records.

### 5.3 Node records

Supply exactly `nnode` records:

```text
node_id x y z
```

Node identifiers must be consecutive and ordered from 1. Coordinates must be
finite.

### 5.4 Beam-element records

Supply exactly `nelem` records:

```text
element_id material_id endpoint_1 endpoint_2
```

For both Euler–Bernoulli and Timoshenko beams, supply only the two endpoints:

```text
element_id material_id endpoint_1 endpoint_2
```

For Euler–Bernoulli beams, OpenSG creates one midpoint and uses a three-node
Hermite interpolation. For Timoshenko beams, OpenSG creates nodes at one-third
and two-thirds of the member length and uses a four-node Lagrange interpolation.
The supplied endpoints are the nodes used by periodic and junction records.

Element identifiers must be consecutive and ordered from 1. An element cannot
repeat a node or reference an undefined node or material.

### 5.5 Beam-orientation records

If `trans_flag=1`, supply one orientation record for every beam:

```text
element_id x1 y1 z1 x2 y2 z2 x3 y3 z3
```

The three points define the local frame:

1. point 1 is the orientation origin;
2. the vector from point 1 to point 2 defines local axis 1;
3. point 3 defines the local 1–2 plane.

The third point must not lie on the beam axis. For circular sections, changing
the transverse direction should not change the physical result, but consistent
orientations are still recommended.

If `trans_flag=0`, OpenSG constructs a transverse direction automatically.

### 5.6 Periodic node records

Supply exactly `nslave` records:

```text
slave_node master_node
```

The fluctuating beam variables of the two endpoint nodes are identified. Opposite-face,
edge, and corner nodes may be linked through several records to one periodic
representative.

For a three-dimensional SG, the periodic node translations and junction image
shifts together must span three independent directions.

### 5.7 Material records and SG volume

The currently supported solid-junction material is linear isotropic elasticity.
The standard one-property-set record is:

```text
material_id 0 1
0.0 1.0
E nu
```

The final scalar record is the physical volume of the periodic SG box:

```text
SG_volume
```

For an SG with dimensions (10\times10\times10) mm:

```text
1 0 1
0.0 1.0
160000.0 0.28

1000.0
```

The volume is the periodic box volume, not the material volume.

## 6. Beam and junction supplement: `model.sc.msg`

### 6.1 Header

```text
version nbeamtype nbeamassign njunctiontype njunction nconnection jtrans_flag
```

The current version is 1. The remaining entries give the number of records in
each following section. If `jtrans_flag=1`, one orientation record is required
for every junction instance.

Each beam type begins with:

```text
beam_type_id theory interpolation_node_count nstrain
```

The interpolation-node count describes the element created inside OpenSG; it
does not add nodes to the main `.sc` connectivity. Use 3 for Euler–Bernoulli
and 4 for Timoshenko.

### 6.2 Euler–Bernoulli beam type

The beam-type header is:

```text
beam_type_id 0 3 4
```

It is followed by four rows of the complete section-stiffness matrix. The
generalized-strain order is:

```text
extension, twist, bending about local 2, bending about local 3
```

For an uncoupled section:

```text
1 0 3 4
EA  0   0    0
0   GJ  0    0
0   0   EI2  0
0   0   0    EI3
```

### 6.3 Timoshenko beam type

The beam-type header is:

```text
beam_type_id 1 4 6
```

It is followed by six rows of the complete section-stiffness matrix. The order
is:

```text
extension, shear 12, shear 13, twist, bending about local 2,
bending about local 3
```

For an uncoupled section:

```text
1 1 4 6
EA  0    0    0   0    0
0   GA2  0    0   0    0
0   0    GA3  0   0    0
0   0    0    GJ  0    0
0   0    0    0   EI2  0
0   0    0    0   0    EI3
```

Any desired shear
correction is included directly in `GA2` and `GA3`.

### 6.4 Beam assignments

After all beam-type matrices, supply exactly `nbeamassign` records:

```text
element_id beam_type_id
```

Every beam must be assigned once. Undefined, duplicate, missing, and unused
identifiers are rejected.

## 7. Junction flag 0: rigid shared-node junctions

A flag-0 supplement contains no junction records. For one beam type and
`nelem` beams, its header is:

```text
1 1 nelem 0 0 0 0
```

Beams that meet rigidly use the same endpoint node. For example:

```text
1 1 10 11
2 1 10 12
3 1 10 13
```

All three beams share node 10. Do not define junction types, solid junctions,
or `.kj` sources when `junction_flag=0`.

Periodic boundary nodes must be identified by the slave/master records in the
main `.sc` file.

## 8. Junction flag 1: calculate junction stiffness from a solid model

For flag 1, the supplement continues after the beam assignments with junction
types, junction instances, optional junction orientations, and connections.

### 8.1 Junction types

```text
junction_type_id nconnectionpoint source.sc
```

Example:

```text
1 10 junction_1.sc
2 12 junction_2.sc
3 5  junction_3.sc
```

One solid model is analyzed for each junction type, even if that type is used
by several instances.

### 8.2 Junction instances

```text
junction_id junction_type_id x y z
```

The last three values are the junction origin in SG coordinates.

### 8.3 Junction orientations

If `jtrans_flag=1`, supply one three-point orientation record for each junction:

```text
junction_id x1 y1 z1 x2 y2 z2 x3 y3 z3
```

If `jtrans_flag=0`, the junction type axes and SG axes are identical.

### 8.4 Junction-to-beam connections

```text
junction_id connection_point_id element_id endpoint shift_x shift_y shift_z
```

`endpoint` must be 1 or 2. A zero shift is used for an ordinary internal
connection:

```text
3 1 23 1 0.0 0.0 0.0
```

A periodic boundary connection uses a shift:

```text
1 6 6 1 10.0 0.0 0.0
```

The beam endpoint in this record is the **connection point**. The endpoint plus
the shift must coincide with the transformed junction connection-point
position. Each connection is listed once, and one beam endpoint cannot belong
to two junction connection points.

### 8.5 Boundary junction input

Junctions may be located on an SG face, edge, or corner. Enter a boundary
junction in the same way as an internal junction:

1. enter its junction-instance record using its SG coordinates;
2. enter one record for every junction connection point;
3. use a zero shift when the beam endpoint already coincides with the connection point;
4. otherwise, enter the periodic translation that makes the endpoint coincide
   with the connection point.

OpenSG handles the periodic mapping through the beam endpoint and connection
records. The user does not apply periodic constraints to nodes in the local 3D
solid-junction mesh, and no special junction type is required for a face, edge,
or corner location.

If two locations on opposite SG boundaries represent the same periodic
junction, list one junction instance and map its connection points using the
shift columns. This prevents the same junction from being entered twice.

## 9. Solid-junction input for flag 1

Each junction type requires:

```text
junction_name.sc
junction_name.sc.msg
```

### 9.1 Solid-junction `.sc` file

The control record is:

```text
0 0 0 0
```

The mesh-control record is:

```text
3 nnode nsolid nmaterial 0 0
```

Node records use the standard format:

```text
node_id x y z
```

OpenSG accepts the solid tetrahedral element types already represented by its
standard `.sc` mesh input. A linear tetrahedron is written as:

```text
element_id material_id n1 n2 n3 n4
```

A quadratic tetrahedron is written as:

```text
element_id material_id n1 n2 n3 n4 0 n12 n23 n31 n14 n24 n34
```

The zero after the four corner nodes is part of the standard solid-element
record. The six remaining nodes are the midside nodes on edges 12, 23, 31, 14,
24, and 34.

The file ends with the isotropic material records and the solid mesh volume.

### 9.2 Solid-junction connection-point and interface file

The first record of `junction_name.sc.msg` is:

```text
version nconnectionpoint
```

Each connection point begins with:

```text
connection_point_id nface x y z b11 b12 b13 b21 b22 b23 b31 b32 b33
```

Here `(x,y,z)` is the connection-point origin in the junction-type coordinate
system. The three rows of `b` are the local connection-point axes expressed in
the junction-type coordinate system.

For a TET4 mesh, the connection-point header is followed by `nface` linear
triangular interface faces:

```text
n1 n2 n3
```

For a TET10 mesh, use quadratic triangular interface faces:

```text
n1 n2 n3 n12 n23 n31
```

Each interface must consistently use TRI3 or TRI6 records matching its
adjacent solid elements. Interface faces must be exterior solid faces and must
lie in the plane defined at the corresponding connection point. Different
interfaces may meet along perimeter nodes, but the same face cannot belong to
two interfaces.

A complete linear example is provided by
[`linear_two_connection_cube_junction.sc`](../examples/beam_hybrid/linear_two_connection_cube_junction.sc)
and its companion `.sc.msg` file. The BCCH flag-1 examples demonstrate the
quadratic TET10/TRI6 form.

## 10. Junction flag 2: read a reusable junction stiffness

Flag 2 uses the same SG nodes, beams, junction instances, and connections as
flag 1. Only the junction-type source changes:

```text
junction_type_id nconnectionpoint source.kj
```

Example:

```text
1 10 junction_1.sc.kj
2 12 junction_2.sc.kj
3 5  junction_3.sc.kj
```

### 10.1 `.kj` file

The first record is:

```text
version nconnectionpoint
```

Each connection point is described by:

```text
connection_point_id x y z b11 b12 b13 b21 b22 b23 b31 b32 b33
```

The connection-point records are followed by the complete junction stiffness matrix. Its
size is:

```text
(6*nconnectionpoint) rows by (6*nconnectionpoint) columns
```

For example, a junction with three connection points requires an (18\times18)
matrix, and a junction with twelve connection points requires a (72\times72)
matrix.

Do not change the connection-point order, origins, or frames without
transforming the stiffness matrix consistently.

### 10.2 Same-basename solid input for dehomogenization

A flag-2 homogenization reads only the `.kj` file. A flag-2
dehomogenization also reads the already prepared solid mesh with the same
basename:

```text
junction.sc.kj  -> junction.sc and junction.sc.msg
junction.kj     -> junction.sc and junction.sc.msg
```

The solid input uses the format in Section 9. Its connection identifiers,
origins, frames, and order must match the `.kj` connection records. OpenSG
stops with an input error if the `.sc` or `.sc.msg` file is absent or the
connection geometry differs. Neither homogenization nor dehomogenization
creates the junction mesh.

## 11. BCCH tutorial examples

The directory
[`examples/beam_hybrid/bcch_lambda06`](../examples/beam_hybrid/bcch_lambda06)
contains six complete BCCH examples for \(\lambda=0.6\):

| File | Beam theory | Junction flag |
|---|---|---:|
| `bcch_euler_flag0.sc` | Euler–Bernoulli | 0 |
| `bcch_euler_flag1.sc` | Euler–Bernoulli | 1 |
| `bcch_euler_flag2.sc` | Euler–Bernoulli | 2 |
| `bcch_timoshenko_flag0.sc` | Timoshenko | 0 |
| `bcch_timoshenko_flag1.sc` | Timoshenko | 1 |
| `bcch_timoshenko_flag2.sc` | Timoshenko | 2 |

The same directory contains eight solid-junction inputs and their eight `.kj`
files. The junctions have 10, 12, 5, 5, 3, 3, 3, and 3 connection points.

### 11.1 First run: rigid Euler beams

From the OpenSG source directory:

```text
opensg examples/beam_hybrid/bcch_lambda06/bcch_euler_flag0.sc 3D H
```

Expected effective Young's modulus:

```text
E1 = 5.3830218E+01 MPa
```

### 11.2 Read existing junction stiffness

This is the quickest hybrid example:

```text
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag2.sc 3D H
```

Expected effective Young's modulus:

```text
E1 = 7.0389868E+01 MPa
```

### 11.3 Calculate junction stiffness from solid meshes

```text
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag1.sc 3D H
```

This analyzes all eight solid-junction types, writes eight `.kj` files, and
then performs SG homogenization. It should reproduce the flag-2 result to
roundoff.

### 11.4 Run all six examples

```text
opensg examples/beam_hybrid/bcch_lambda06/bcch_euler_flag0.sc 3D H
opensg examples/beam_hybrid/bcch_lambda06/bcch_euler_flag1.sc 3D H
opensg examples/beam_hybrid/bcch_lambda06/bcch_euler_flag2.sc 3D H
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag0.sc 3D H
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag1.sc 3D H
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag2.sc 3D H
```

The rerun results are:

| Beam | Flag | (E_1) (MPa) | Relative (E_1) difference from SwiftComp |
|---|---:|---:|---:|
| Euler | 0 | 53.830218 | -23.5539% |
| Euler | 1 | 72.653750 | +3.1780% |
| Euler | 2 | 72.653750 | +3.1780% |
| Timoshenko | 0 | 52.121897 | -25.9800% |
| Timoshenko | 1 | 70.389868 | -0.0370% |
| Timoshenko | 2 | 70.389868 | -0.0370% |

The refined SwiftComp value is (E_1=70.415945) MPa. The relative (E_1)
difference is calculated as (100(E_1/E_1^{\mathrm{SwiftComp}}-1)). Flags 1
and 2 produce identical matrices for a given beam theory because flag 2 reads
the same junction stiffness produced by flag 1.

For both junction treatments, Euler--Bernoulli is slightly stiffer than
Timoshenko because it neglects transverse shear deformation. The flag-0 rows
represent the simplified pure-beam, rigid-junction model rather than the
resolved solid geometry. The closest validated BCCH comparison is the
four-node Timoshenko model with flags 1 and 2.

## 12. Reading the result file

The effective stiffness is printed in engineering-strain Voigt order:

```text
11, 22, 33, 23, 13, 12
```

Thus:

```text
C11 C12 C13 C14 C15 C16
C21 C22 C23 C24 C25 C26
C31 C32 C33 C34 C35 C36
C41 C42 C43 C44 C45 C46
C51 C52 C53 C54 C55 C56
C61 C62 C63 C64 C65 C66
```

The compliance matrix is the inverse or generalized inverse of the stiffness
matrix. The reported engineering constants are (E_1,E_2,E_3,G_{12},G_{13},
G_{23},\nu_{12},\nu_{13},\nu_{23}).

The result file reports both the number of endpoint nodes read from the input
and the total number of nodes after OpenSG creates the interpolation nodes.

`Zero-energy mechanism = true` means that at least one effective stiffness mode
is zero or numerically negligible. This can be intentional, but it commonly
indicates missing members, missing periodic constraints, or incorrect
connections.

## 13. Dehomogenization

Dehomogenization recovers three-dimensional displacement, engineering strain,
and Cauchy stress from a prescribed macroscopic state. The solver uses only
the model and recovery inputs described below. Mesh-generation and plotting
programs are separate preprocessing and post-processing utilities.

### 13.1 Required inputs

A dehomogenization run requires:

```text
model.sc
model.sc.msg
model.sc.glb
```

Every beam type must have a `BEAM_RECOVERY` record referencing its VABS
cross-section source. Junction flag 1 reuses each junction type's solid source.
For junction flag 2, the same-basename solid `.sc/.sc.msg` files described in
Section 10.2 must already exist.

The `.k` homogenization output is not an input to dehomogenization. Users may
run `3D H` and `3D L` independently from the same model files.

### 13.2 Beam-recovery records

After all junction-connection records in `model.sc.msg`, associate every beam
type with its VABS section input:

```text
BEAM_RECOVERY beam_type_id vabs_section_source
```

Example:

```text
BEAM_RECOVERY 1 sections/square0.sg
BEAM_RECOVERY 2 sections/tube.sg
```

Relative paths are interpreted relative to `model.sc.msg`. The VABS executable
is selected by `--vabs PATH`, the `VABS_EXE` environment variable, or the
system `PATH`, in that order. The standard Windows VABS installation path is a
final fallback.

### 13.3 Global-field file: `model.sc.glb`

The elastic three-dimensional global-field file follows the SwiftComp layout:

```text
v1 v2 v3
C11 C12 C13
C21 C22 C23
C31 C32 C33
id1
q1 q2 q3 q4 q5 q6
```

`v1,v2,v3` are the macroscopic displacement components and `C` is the
deformation matrix. The final flag and vector select the prescribed quantity:

```text
id1 = 1: q = [e11,e22,e33,2e23,2e13,2e12]
id1 = 0: q = [s11,s22,s33,s23,s13,s12]
```

For `id1=1`, OpenSG calculates stress from the effective stiffness. For
`id1=0`, it calculates strain from the effective compliance.

### 13.4 Beam recovery

OpenSG first recovers each beam's centerline displacement, rotation,
generalized strain, and resultant in its local frame. It then calls VABS at
the requested axial stations to obtain cross-section nodal displacement,
strain, and stress.

For Timoshenko beams, extension, two transverse shears, twist, and two bending
components are recovered directly. For Euler--Bernoulli beams, extension,
twist, and bending components are recovered from the Hermite field; transverse
forces are obtained from the axial bending-moment gradients before VABS
recovery. Both formulations therefore produce complete three-dimensional
local fields.

### 13.5 Junction recovery

For junction flag 1, the supplied solid-junction mesh used during
homogenization is reused. For junction flag 2, OpenSG reads the same-basename
solid input associated with the `.kj` source. The generalized connection
displacements are applied to that solid model, and OpenSG recovers nodal
displacement, engineering strain, and Cauchy stress.

The recovery solid currently supports TET4/TRI3 and TET10/TRI6. Its connection
geometry must match the junction stiffness file. The solver never calls a
Boolean operation or creates a solid mesh.

### 13.6 Local output files

`model.sc.u` contains one displacement record per recovered node:

```text
node_number u1 u2 u3
```

`model.sc.sn` contains coordinates followed by strain and stress:

```text
y1 y2 y3 e11 e22 e33 2e23 2e13 2e12 s11 s22 s33 s23 s13 s12
```

All vectors and tensors are written in the global SG coordinate system. Beam
and junction records are combined in these files with consecutive output node
numbers.

### 13.7 Example post-processing

The simple-cubic example separates solving from plotting. Its verification
step runs Euler and Timoshenko through the public homogenization and
dehomogenization interfaces, then writes pointwise OpenSG and SwiftComp
centerline data. Its VABS section, global fields, and SwiftComp reference data
are stored inside the example directory. Run the comparison and produce the
six separate stress-component figures afterward with:

```text
python examples/simple_cubic_dehomogenization/verify_dehomogenization.py
python examples/simple_cubic_dehomogenization/plot_centerline.py
```

This plotting program is not imported by either solver.

### 13.8 Reusable Python API

The same workflow is available without the command-line dispatcher:

```python
from pathlib import Path
from fe_jax import (
    dehomogenize,
    homogenize,
    read_global_fields,
    write_local_outputs,
)

input_path = Path("model.sc")
model, supplement, result = homogenize(input_path)
global_fields = read_global_fields(
    Path(str(input_path) + ".glb"),
    result.effective_stiffness,
    result.effective_compliance,
)
local_fields = dehomogenize(
    model,
    supplement,
    result,
    global_fields,
    stations=3,
    executable="/path/to/VABS",
)
write_local_outputs(input_path, local_fields)
```

All paths come from the caller or the supplied input files. The API does not
load any example geometry or generate a junction mesh.

## 14. Input checks and common errors

OpenSG validates the input before homogenization and reports the item that must
be corrected.

### Wrong junction flag

```text
OpenSG input error: junction_flag=0 selects a pure-beam model, but the
supplement defines junction records.
```

Remove the junction records or select flag 1 or 2.

### Incomplete periodicity

```text
OpenSG input error: The periodic node shifts and junction image shifts span
only 2 independent directions; a 3D periodic SG requires three.
```

Check missing opposite-face pairs and boundary-junction image shifts.

### Incorrect connection-point number

```text
OpenSG input error: Junction 1 requires connection points 1 through 2;
missing=[1], invalid=[3].
```

Check the connection section against the connection-point order in the solid
or `.kj` source.

### Beam endpoint does not meet the connection point

The beam endpoint, shift, junction origin, orientation, and connection-point origin
must describe the same physical point. Check all five items if a coincidence
error is reported.

### Invalid solid interface

If an interface is not planar, is not on the exterior of the solid mesh, or
uses face connectivity inconsistent with the adjacent solid element, the
solid-junction analysis stops before solving.

### Missing flag-2 recovery mesh

If `source.sc.kj` is referenced for a flag-2 junction, dehomogenization expects
`source.sc` and `source.sc.msg` in the same directory. Generate or copy those
inputs before running `3D L`; OpenSG does not create them during a solver run.

### Missing beam recovery or VABS

Every beam type used during dehomogenization requires one `BEAM_RECOVERY`
record. Check that the referenced section file exists and that VABS can be
found through `--vabs`, `VABS_EXE`, or the standard installation path.

### Invalid global-field input

An elastic 3D `.glb` file must contain exactly the displacement vector, nine
deformation-matrix entries, `id1`, and six strain or stress entries. Check the
component order in Section 13.3.

### Invalid junction stiffness

A `.kj` matrix must have the declared dimensions, be symmetric, contain six
rigid modes, and contain no negative deformational modes. A mismatch usually
means that the `.kj` file was paired with the wrong connection-point geometry
or order.

## 15. Recommended model-preparation procedure

1. Choose one consistent unit system.
2. Define the periodic SG box and its physical volume.
3. Choose Euler–Bernoulli or Timoshenko beams.
4. Define only the two endpoints, connectivity, and local orientation of each
   beam. OpenSG creates all interpolation nodes.
5. For flag 0, merge every rigid junction to one shared node and define all
   periodic node pairs.
6. For flags 1 and 2, give every junction connection its own beam endpoint node.
7. Enter boundary junctions with the same instance and connection records used
   for internal junctions. Do not repeat the same periodic junction on the
   opposite boundary.
8. Use the shift columns when a connection uses a beam endpoint
   represented on another periodic boundary.
9. Check that node-pair and image-shift translations span three directions.
10. For flag 1, prepare TET4/TRI3 or TET10/TRI6 solid-junction meshes.
11. For flag 2, verify that each `.kj` file uses exactly the declared connection
    positions, frames, and order. For dehomogenization, retain its
    same-basename `.sc/.sc.msg` solid input.
12. Add one `BEAM_RECOVERY` VABS section source for every beam type that will
    be dehomogenized.
13. Prepare `model.sc.glb` using the component order in Section 13.3.
14. Run `3D H` and inspect the `.ech` file before accepting the `.k` result.
15. Check stiffness symmetry, positive eigenvalues, the mechanism indicator,
    and expected material/geometry symmetries.
16. Run `3D L`, then inspect `.u` and `.sn` or use a separate plotting program.

## 16. Preparing a new model from an example

For a first user model, copy the example that is closest to the desired
analysis:

- copy `bcch_euler_flag0.sc` and its `.msg` for rigid Euler beams;
- copy `bcch_timoshenko_flag0.sc` and its `.msg` for rigid Timoshenko beams;
- copy a flag-1 pair and the solid junction files when junction stiffness must
  be calculated;
- copy a flag-2 pair and the `.kj` files when reusable junction data are
  available.

Change the files in this order:

1. counts in the `.sc` and `.sc.msg` headers;
2. nodes and beam connectivity;
3. orientations and periodic pairs;
4. section stiffness and beam assignments;
5. junction types and instances;
6. junction connections and shifts;
7. material records and SG volume.

Run after each small change. Input errors should then point directly to the
section most recently modified.
