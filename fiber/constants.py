"""Physical constants for the fiber package (CODATA values from scipy).

The laser models in core/ use c = 3e8. The fiber package needs the exact value:
absolute propagation constants (~6e6 rad/m) are differenced between modes, and a
0.07 % error in c shifts beta by ~4e3 rad/m, the same order as the spacing between
GRIN mode groups.
"""
from scipy import constants as _sc

c = _sc.c
h = _sc.h
hbar = _sc.hbar
kB = _sc.k
