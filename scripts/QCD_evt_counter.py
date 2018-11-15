import h5py
from glob import glob
import numpy as np
import os

folder = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23/qcd'

files = glob(folder+'/*.h5')

print len(files)

broken_files_file = open(folder+'/broken_files.txt', 'w')
N_tot = 0
errors = 0

for i, fname in enumerate(files):
        if i%100 == 0:
            print 'At file', i, 'size:', N_tot, 'errors:', errors
        try:
            f = h5py.File(fname, 'r')
            N_tot += f['HLF'].shape[0]
        except:
            broken_files_file.write(fname)
            errors += 1
print 'Total of {} events in {} files'.format(N_tot, len(files))
broken_files_file.write('Total of {} events in {} files'.format(N_tot, len(files)))
broken_files_file.close()
