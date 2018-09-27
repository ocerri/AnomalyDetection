import h5py
from glob import glob
import numpy as np
import os
import argparse

inpath = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23/'
outpath = '/afs/cern.ch/user/o/ocerri/cernbox/AnomalyDetection/data/HLF_ONLY/'
SM_labels = ['qcd_lepFilter_13TeV_HLFONLY', 'qcd_lepFilter_13TeV_HLFONLY2', 'ttbar_lepFilter_13TeV_HLFONLY', 'Wlnu_lepFilter_13TeV_HLFONLY',]
BSM_labels = ['leptoquark_LOWMASS_lepFilter_13TeV', 'Wprime_LOWMASS_lepFilter_13TeV', 'Zprime_LOWMASS_lepFilter_13TeV', 'Ato4l_lepFilter_13TeV']

parser = argparse.ArgumentParser()
parser.add_argument("sample_label", type=str, help='Name of the sample', nargs='+')
parser.add_argument('-N', "--MaxNumber", type=int, default=999999999999999, help='Max number of events')
parser.add_argument('-i', "--input_path", type=str, default=inpath)
parser.add_argument('-o', "--output_path", type=str, default=outpath)
parser.add_argument('-F', "--force", action='store_true', default=outpath)
args = parser.parse_args()


if len(args.sample_label) == 1:
    if args.sample_label[0] == 'SM':
        args.sample_label = SM_labels
    if args.sample_label[0] == 'BSM':
        args.sample_label = BSM_labels

print 'Running on:'
print args.sample_label
print

for sample_label in args.sample_label:
    outname = args.output_path+sample_label+'_sample.npy'
    if os.path.isfile(outname):
        if args.force:
            os.remove(outname)
        else:
            print 'File '+outname+' already existing'
            exit(0)

    hlf_train = np.zeros((0,23))

    file_list = glob(inpath + sample_label +'/*.h5')
    print sample_label
    print len(file_list)
    errors = 0
    for i, fname in enumerate(file_list):
        if i%100 == 0 or i == len(file_list)-1:
            print 'At file', i, 'size:', hlf_train.shape[0], 'errors:', errors
        try:
            f = h5py.File(fname, 'r')
            hlf = np.array(f['HLF'])[:, 1:]

            #Remove Sphericity
            if hlf.shape[1] == 24 and list(f['HLF_Names'])[5] == 'SPH':
                hlf = np.delete(hlf, 4, 1)
            elif hlf.shape[1] != 23:
                print 'Non matching shapes ---> Exiting'
                exit(0)

            hlf_train = np.concatenate((hlf_train, hlf))
        except:
            errors += 1
            if errors < 20:
                print '[{}]'.format(errors), fname, 'failed'
            elif errors > 100:
                print 'Too many errors'
                exit(0)

        if hlf_train.shape[0] > args.MaxNumber:
            print 'Max number of {} overcome'
            print 'At file', i, 'size:', hlf_train.shape[0], 'errors:', errors
            break


    print hlf_train.shape

    if hlf_train.shape[0] > 0:
        np.save(outname, hlf_train)
    else:
        print 'File empty. No output.'
