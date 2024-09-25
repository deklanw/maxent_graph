from .bicm import BICM as BICM
from .biecm import BIECM as BIECM
from .bwcm import BWCM as BWCM
from .decm import DECM as DECM
from .ecm import ECM as ECM
from .rcm import RCM as RCM

import jax

# ensure jax is using doubles. important.
jax.config.update("jax_enable_x64", True)
