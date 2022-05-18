from figaro.mixture import DPGMM

# These instructions are redundant (they are made while importing from figaro.mixture)
from figaro.exceptions import except_hook
import sys
sys.excepthook = except_hook

# Improper FIGARO usage
mix = DPGMM([-1,1])
mix.add_new_point(5)
