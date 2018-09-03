import h5py
from glob import glob
import numpy as np
import os
import argparse

inpath = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23/'
outpath = '/afs/cern.ch/user/o/ocerri/cernbox/AnomalyDetection/data/'
SM_labels = ['Wlnu_lepFilter_13TeV', 'qcd_lepFilter_13TeV', 'ttbar_lepFilter_13TeV']
BSM_labels = ['AtoChHW_lepFilter_13TeV', 'AtoChHW_HIGHMASS_lepFilter_13TeV', 'Ato4l_lepFilter_13TeV', 'Wprime_lepFilter_13TeV', 'Zprime_lepFilter_13TeV']

parser = argparse.ArgumentParser()
parser.add_argument("sample_label", type=str, help='Name of the sample')
parser.add_argument('-N', "--MaxNumber", type=int, default=3000000, help='Max number of events')
parser.add_argument('-i', "--input_path", type=str, default=inpath)
parser.add_argument('-o', "--output_path", type=str, default=outpath)
parser.add_argument('-F', "--force", action='store_true', default=outpath)
args = parser.parse_args()

for l in BSM_labels:
    args.sample_label = l

    outname = args.output_path+args.sample_label+'_sample.npy'
    if os.path.isfile(outname):
        if args.force:
            os.remove(outname)
        else:
            print 'File already existing'
            exit(0)

    hlf_train = np.zeros((0,24))

    file_list = glob(inpath + args.sample_label +'/*.h5')
    print args.sample_label
    print len(file_list)
    errors = 0
    for i, fname in enumerate(file_list):
        if i%50 == 0:
            print 'At file', i, 'size:', hlf_train.shape[0]
        try:
            f = h5py.File(fname, 'r')
            hlf = np.array(f['HLF'])[:, 1:]
            hlf_train = np.concatenate((hlf_train, hlf))
        except:
            errors += 1
            print '[{}]'.format(errors), fname, 'failed'

        if hlf_train.shape[0] > args.MaxNumber:
            break


    print hlf_train.shape

    np.save(outname, hlf_train)
