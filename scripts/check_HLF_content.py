import h5py
from glob import glob
import numpy as np
import os

folder = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23'
to_check = ['Ato4l_lepFilter_13TeV', 'leptoquark_LOWMASS_lepFilter_13TeV', 'qcd_lepFilter_13TeV_HLFONLY', 'qcd_lepFilter_13TeV_HLFONLY2',
            'ttbar_lepFilter_13TeV_HLFONLY', 'Wlnu_lepFilter_13TeV_HLFONLY', 'Wprime_LOWMASS_lepFilter_13TeV', 'Zprime_LOWMASS_lepFilter_13TeV'
            ]


files = glob(folder+'/*.h5')

for pname in to_check:
    fname = glob(folder+'/'+pname+'/*.h5')[0]
    print os.path.basename(fname)
    f = h5py.File(fname, 'r')
    print f.keys()
    print f['HLF'].shape
    try:
        aux = f['HLF_Names']
    except:
        aux = f['HLF_names']
    print aux.shape
    print list(aux)
    print '\n\n'

# broken_files_file.write('Total of {} events in {} files'.format(N_tot, len(files)))
# broken_files_file.close()
