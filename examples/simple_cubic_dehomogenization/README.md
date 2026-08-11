# Simple-cubic dehomogenization comparison

The solver verification writes pointwise Euler, Timoshenko, and SwiftComp
centerline values without importing any plotting package:

```powershell
python examples\simple_cubic_dehomogenization\verify_dehomogenization.py
```

Plotting is a separate example post-processing step:

```powershell
python examples\simple_cubic_dehomogenization\plot_centerline.py
```

The junction portion uses actual C3D20 nodes at the SwiftComp centerline
coordinates. The beam portion uses VABS cross-section-center recovery.
