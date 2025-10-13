#!/usr/bin/env python3

import sys
import basis_set_exchange as bse

flist = [
    'aug-cc-pV5Z', 'aug-cc-pV5Z-PP', 'aug-cc-pVDZ', 'aug-cc-pVDZ-PP',
    'aug-cc-pVQZ', 'aug-cc-pVQZ-PP', 'aug-cc-pVTZ', 'aug-cc-pVTZ-PP',
    'aug-cc-pwCV5Z-PP', 'aug-cc-pwCVDZ-PP', 'aug-cc-pwCVQZ-PP',
    'aug-cc-pwCVTZ-PP', 'cc-pCVDZ-F12', 'cc-pCVQZ-F12', 'cc-pCVTZ-F12',
    'cc-pVDZ-F12', 'cc-pVQZ-F12', 'cc-pVTZ-F12'
]

for bs in flist:
    f = bs + '.metadata.json'
    a = bse.fileio.read_json_basis(f)
    a['auxiliaries']['optri'] = bs.lower() + '-optri'
    bse.fileio.write_json_basis(f, a)
