#!/usr/bin/env python3

import os
import sys
import basis_set_exchange as bse

meta, table, _, _ = bse.fileio.get_all_filelist(bse.get_data_dir())

for t in table:
    subdirs = set()
    td = bse.fileio.read_json_basis(t)
    for v in td['elements'].values():
        subdirs.add(os.path.split(v)[0])

    if len(subdirs) > 1:
        print(
            "File: {} has more than one subdir. You are on your own".format(t))
        continue
    assert len(subdirs) > 0
    subdir = subdirs.pop()

    # Find metadata file
    tbase = t.split('.')[0]

    matching_meta = [x for x in meta if x.startswith(tbase + '.')]
    if len(matching_meta) != 1:
        raise RuntimeError("matching meta != 1")
    matching_meta = matching_meta[0]
    print(tbase)
    print('   into ' + subdir)
    print(t)
    print('   ' + matching_meta)
