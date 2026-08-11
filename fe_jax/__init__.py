from importlib.util import find_spec

_jax_modules = ("jax", "jaxopt", "numba", "igl", "basix")
if all(find_spec(name) is not None for name in _jax_modules):
    import jax

    jax.config.update("jax_enable_x64", True)
    from .np_types import *
    from .basis_quadrature import *
    from .fea import *
    from .linear_elasticity import *
    from .hyperelasticity import *
    from .setup import *
    from .utils import *
    from .sc_to_msh import *

from .beam import BeamElement, BeamTheory, BeamType
from .dehomogenization import BeamStationState, LocalFields, dehomogenize
from .hybrid_homogenization import HomogenizationResult, homogenize
from .junction import JunctionStiffness, read_junction_stiffness, write_junction_stiffness
from .sc_glb_input import GlobalFields, read_global_fields
from .sc_local_output import write_local_outputs
#from .periodic_dofmap import *
#from .multiscale import *
#from .fiber_mechanics import *
