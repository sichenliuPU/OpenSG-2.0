# BCCH lambda=0.6 examples

This directory contains six complete OpenSG inputs:

Every beam element in these inputs lists only its two physical endpoints.
OpenSG creates the Euler midpoint or the two Timoshenko interior nodes during
the analysis.

| Main input | Beam theory | Junction flag |
|---|---|---:|
| `bcch_euler_flag0.sc` | Euler–Bernoulli | 0 |
| `bcch_euler_flag1.sc` | Euler–Bernoulli | 1 |
| `bcch_euler_flag2.sc` | Euler–Bernoulli | 2 |
| `bcch_timoshenko_flag0.sc` | Timoshenko | 0 |
| `bcch_timoshenko_flag1.sc` | Timoshenko | 1 |
| `bcch_timoshenko_flag2.sc` | Timoshenko | 2 |

Run any case from the OpenSG source directory, for example:

```text
opensg examples/beam_hybrid/bcch_lambda06/bcch_timoshenko_flag2.sc 3D H
```

Files `junction_1.sc` through `junction_8.sc` and their companion `.msg`
files are the TET10 sources used by flag 1. Files ending in `.sc.kj` are the
reusable junction stiffness inputs used by flag 2.

The validated comparison with the refined SwiftComp solid model is the
Timoshenko model with junction flag 1 or 2: `E1 = 70.389868 MPa`, compared
with `70.415945 MPa` from SwiftComp (a `-0.0370%` difference). The corresponding
Euler result is slightly stiffer, `E1 = 72.653750 MPa`, because it neglects
transverse shear deformation. The flag-0 models use simplified rigid beam
junctions and are not resolved-solid BCCH predictions.

The full input specification and tutorial are in
[`../../../docs/OpenSG_Beam_Junction_User_Manual.md`](../../../docs/OpenSG_Beam_Junction_User_Manual.md).
