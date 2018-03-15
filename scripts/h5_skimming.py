import h5py
from glob import glob
import numpy as np
import os

inpath = '/eos/cms/store/cmst3/group/dehep/TOPCLASS/REDUCED_IsoLep/'

outpath = '/afs/cern.ch/user/o/ocerri/cernbox/AnomalyDetection/data/'

SM_labels = ['qcd_lepFilter_13TeV', 'ttbar_lepFilter_13TeV', 'wjets_lepFilter_13TeV']
BSM_labels = ['AtoChHW', 'AtoChHW_HIGHMASS']

N_evnt_train = 70000
hlf_train = np.zeros((0,14))
label_train = np.zeros((0,))

val_split = 0.7

hlf_val = np.zeros((0,14))
label_val = np.zeros((0,))

for k, lab in enumerate(BSM_labels):
    file_list = glob(inpath + lab +'/*.h5')
    print lab
    print len(file_list)
    errors = 0
    for i, fname in enumerate(file_list):
        if i%100 == 0:
            print 'At file', i, 'size:', hlf_train.shape[0]
        try:
            f = h5py.File(fname, 'r')
            hlf = np.array(f['HLF'])[:, 1:]
            i_sep = int(hlf.shape[0]*val_split)

            hlf_train = np.concatenate((hlf_train, hlf[:i_sep]))
            hlf_val = np.concatenate((hlf_val, hlf[i_sep:]))
        except:
            errors += 1
            print '[{}]'.format(errors), fname, 'failed'

        if hlf_train.shape[0]>(k+1)*N_evnt_train:
            break

    label_train = np.concatenate((label_train, k*np.ones(hlf_train.shape[0]-label_train.shape[0])))
    label_val = np.concatenate((label_val, k*np.ones(hlf_val.shape[0]-label_val.shape[0])))

    print hlf_train.shape
    print label_train.shape

outname = outpath+'BSM_train.h5'
if os.path.isfile(outname):
    os.remove(outname)
h5f = h5py.File(outname, 'w')
h5f.create_dataset('HLF', data=hlf_train)
h5f.create_dataset('labels', data=label_train)
h5f.close()

outname = outpath+'BSM_val.h5'
if os.path.isfile(outname):
    os.remove(outname)
h5f = h5py.File(outname, 'w')
h5f.create_dataset('HLF', data=hlf_val)
h5f.create_dataset('labels', data=label_val)
h5f.close()
