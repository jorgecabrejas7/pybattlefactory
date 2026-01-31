try:
    # Try to import from the compiled C++ extension
    from .pybattle_native import *
except ImportError:
    # Fallback to direct import if not installed as a package
    try:
        import pybattle as pybattle_native
        from pybattle import *
    except ImportError:
        pass

from .pkmn_env import PokemonEnv
from .factory_hrl_env import FactoryHRL_Env, FactoryPhase
