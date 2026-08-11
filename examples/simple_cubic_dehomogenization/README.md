# Simple-cubic dehomogenization comparison

The solver verification runs both Euler and Timoshenko cases through the public
OpenSG homogenization and dehomogenization APIs. It writes pointwise OpenSG and
SwiftComp centerline values without importing any plotting package:

```powershell
python examples\simple_cubic_dehomogenization\verify_dehomogenization.py
```

Plotting is a separate example post-processing step. It writes six individual
stress figures suitable for inclusion in a manuscript:

```powershell
python examples\simple_cubic_dehomogenization\plot_centerline.py
```

The junction portion uses actual C3D20 nodes at the SwiftComp centerline
coordinates. The beam portion uses VABS cross-section-center recovery.

All numerical inputs used by the comparison are stored under `inputs/` and
`reference/`. The script does not read files outside this example directory.
VABS must be available through `--vabs`, `VABS_EXE`, or the system `PATH`.

To exercise the installed command-line solver for both beam theories:

```powershell
python examples\simple_cubic_dehomogenization\verify_dehomogenization.py --cli-smoke
```
