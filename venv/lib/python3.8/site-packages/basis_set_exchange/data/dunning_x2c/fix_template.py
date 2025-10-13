#!/usr/bin/env python3

import sys
import basis_set_exchange as bse

for f in sys.argv[1:]:
    a = bse.fileio.read_json_basis(f)
    a = bse.manip.prune_basis(a)
    bse.fileio.write_json_basis(f, a)
