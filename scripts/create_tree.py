import numpy as np
import root_numpy as rtnp
import sys

fname = sys.argv[1]

hlf_features = ['HT', 'METp', 'METo', 'MT', 'nJets',
                'bJets', 'allJetMass', 'LepPt', 'LepEta',
                'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge',
                'LepIsEle', 'nMu', 'allMuMass', 'allMuPt', 'nEle',
                'allEleMass', 'allElePt', 'nChHad', 'nNeuHad', 'nPhoton']

b_names = hlf_features

arr = np.load(fname)
arr.dtype = zip(b_names,['<f8']*len(b_names))

rtnp.array2root(arr, fname.replace('.npy', '.root'), treename='T', mode='RECREATE')
