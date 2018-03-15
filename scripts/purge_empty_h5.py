import h5py
from glob import glob
import numpy as np
import os

inpath = '/eos/cms/store/cmst3/group/dehep/TOPCLASS/REDUCED_IsoLep/'


SM_labels = ['qcd_lepFilter_13TeV', 'ttbar_lepFilter_13TeV', 'wjets_lepFilter_13TeV']

for k, lab in enumerate(SM_labels):
    file_list = glob(inpath + lab +'/*.h5')
    print lab
    print len(file_list)
    errors = 0
    for i, fname in enumerate(file_list):
        try:
            f = h5py.File(fname, 'r')
        except IOError:
            errors += 1
            print '[{}] -- '.format(errors), fname, ' --- failed'
            os.remove(fname)
