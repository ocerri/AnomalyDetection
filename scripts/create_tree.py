import numpy as np
import root_numpy as rtnp
import sys

fname = sys.argv[1]

hlf_features = ['HT', 'MET', 'PhiMET', 'MT', 'nJets', 'bJets',
                'allJetMass', 'LepPt', 'LepEta', 'LepPhi', 'LepIsoCh',
                'LepIsoGamma', 'LepIsoNeu', 'LepCharge', 'LepIsEle', 'nMu',
                'allMuMass', 'allMuPt', 'nEle', 'allEleMass', 'allElePt', 'nChHad',
                'nNeuHad', 'nPhoton']

b_names = hlf_features

arr_train = np.column_stack((x_train, l_train, np.ones_like(l_train)))
arr_val = np.column_stack((x_val, l_val, np.zeros_like(l_val)))

arr = np.concatenate((arr_train, arr_val))
arr.dtype = zip(b_names,['<f8']*len(b_names))

rtnp.array2root(arr, '/Users/olmo/cernbox/AnomalyDetection/data/_root/hlf_flat.root', treename='T', mode='RECREATE')
