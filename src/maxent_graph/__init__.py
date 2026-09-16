from .bicm import BICM as BICM
from .biecm import BIECM as BIECM
from .bwcm import BWCM as BWCM
from .dbcm import DBCM as DBCM
from .decm import DECM as DECM
from .ecm import ECM as ECM
from .rcm import RCM as RCM
from .ubcm import UBCM as UBCM

from .counts import BIHPCM as BIHPCM
from .counts import BINBCM as BINBCM
from .counts import BIPCM as BIPCM
from .counts import DHPCM as DHPCM
from .counts import DNBCM as DNBCM
from .counts import DPCM as DPCM
from .counts import DyadLayout as DyadLayout
from .counts import DyadModel as DyadModel
from .counts import UHPCM as UHPCM
from .counts import UNBCM as UNBCM
from .counts import UPCM as UPCM
from .counts import aggregate_blocks as aggregate_blocks
from .counts import from_bicm as from_bicm
from .counts import from_biecm as from_biecm

import jax

# ensure jax is using doubles. important.
jax.config.update("jax_enable_x64", True)
