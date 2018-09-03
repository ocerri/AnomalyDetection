import numpy as np
import h5py
import ROOT as rt

#####

rt.gSystem.Load("/afs/cern.ch/user/m/mpierini/work/DataScience/GenerativeFastSim/Delphes/libDelphes")
#rt.gSystem.Load("/afs/cern.ch/user/m/mpierini/work/TEST/Delphes/libDelphes")
#rt.gSystem.Load("/Users/maurizio/DeepLearning/EXTERNAL/Delphes-3.4.0/libDelphes")
rt.gInterpreter.Declare('#include "/afs/cern.ch/user/m/mpierini/work/DataScience/GenerativeFastSim/Delphes/classes/DelphesClasses.h"')
rt.gInterpreter.Declare('#include "/afs/cern.ch/user/m/mpierini/work/DataScience/GenerativeFastSim/Delphes/external/ExRootAnalysis/ExRootTreeReader.h"')

#####

def PFIso(p, DR, PtMap, subtractPt):
    if p.Pt() <= 0.: return 0.
    DeltaEta = PtMap[:,0] - p.Eta()
    DeltaPhi = PtMap[:,1] - p.Phi()
    pi = rt.TMath.Pi()
    DeltaPhi = DeltaPhi - 2*pi*(DeltaPhi >  pi) + 2*pi*(DeltaPhi < -1.*pi)
    isInCone = DeltaPhi*DeltaPhi + DeltaEta*DeltaEta < DR*DR
    Iso = PtMap[isInCone, 2].sum()/p.Pt()
    if subtractPt: Iso = Iso -1
    return Iso

#####

def ChPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowTrack:
        if h.PT<= 0.5: continue
        pTmap.append([h.Eta, h.Phi, h.PT])
        #nParticles += 1
    #pTmap = np.reshape(pTmap, (nParticles, 3))
    return np.asarray(pTmap)

def NeuPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowNeutralHadron:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
        #nParticles += 1
    #pTmap = np.reshape(pTmap, (nParticles, 3))
    return np.asarray(pTmap)

def PhotonPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowPhoton:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
        #nParticles += 1
    #pTmap = np.reshape(pTmap, (nParticles, 3))
    return np.asarray(pTmap)

#####

def selection(event, TrkPtMap, NeuPtMap, PhotonPtMap, evtID):
    # one electron or muon with pT> 15 GeV
    if event.Electron_size == 0 and event.MuonTight_size == 0: return False, False, False
    foundMuon = None #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0, 0, 1, 1]
    foundEle =  None #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0, 1, 0, 1]
    l = rt.TLorentzVector()
    for ele in event.Electron:        
        if ele.PT <= 25.: continue
        l.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
        pfisoCh = PFIso(l, 0.3, TrkPtMap, True) 
        pfisoNeu = PFIso(l, 0.3, NeuPtMap, False) 
        pfisoGamma = PFIso(l, 0.3, PhotonPtMap, False) 
        if foundEle == None and (pfisoCh+pfisoNeu+pfisoGamma)<0.2: 
            #foundEle.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
            foundEle = [evtID, l.E(), l.Px(), l.Py(), l.Pz(), l.Pt(), l.Eta(), l.Phi(), 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 0, 1, 0, ele.Charge]
    for muon in event.MuonTight:
        if muon.PT <= 25.: continue
        l.SetPtEtaPhiM(muon.PT, muon.Eta, muon.Phi, 0.)
        pfisoCh = PFIso(l, 0.3, TrkPtMap, True)
        pfisoNeu = PFIso(l, 0.3, NeuPtMap, False)
        pfisoGamma = PFIso(l, 0.3, PhotonPtMap, False)
        if foundMuon == None and (pfisoCh+pfisoNeu+pfisoGamma)<0.2: 
            #foundMuon.SetPtEtaPhiM(muon.PT, muon.Eta, muon.Phi, 0.)
            foundMuon = [evtID, l.E(), l.Px(), l.Py(), l.Pz(), l.Pt(), l.Eta(), l.Phi(), 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 0, 0, 1, muon.Charge]
    if foundEle != None and foundMuon != None:
        if foundEle[5] > foundMuon[5]: return True, foundEle, foundMuon
        else: return True, foundMuon, foundEle
    if foundEle != None: return True, foundEle, foundMuon
    if foundMuon != None: return True, foundMuon, foundEle
    return False, None, None

#####

def Convert(filename, minEvt, maxEvt):
    inFile = rt.TFile.Open(filename)
    tree = inFile.Get("Delphes")
    q = rt.TLorentzVector()
    # particles are stored as pT, eta, phi, E, pdgID
    particles = []
    # ['HT', 'MET', 'PhiMET', 'MT', 'nJets', 'nBJets','LepPt', 'LepEta', 'LepPhi', 'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge', 'LepIsEle']
    HLF = []
    Nevt = 0
    Nwritten = 0
    for event in tree:
        if Nevt < minEvt: 
            Nevt += 1
            continue
        if Nevt > maxEvt: continue
        #if Nevt > 100: continue
        #if Nevt > 500: continue
        # isolation maps
        TrkPtMap = ChPtMapp(0.3, event)
        NeuPtMap = NeuPtMapp(0.3, event)
        PhotonPtMap = PhotonPtMapp(0.3, event)
        if TrkPtMap.shape[0] == 0: continue
        if NeuPtMap.shape[0] == 0: continue
        if PhotonPtMap.shape[0] == 0: continue
        selected, lep, otherlep = selection(event, TrkPtMap, NeuPtMap, PhotonPtMap, Nwritten)
        if not selected: 
            Nevt += 1
            continue
        particles.append(lep)
        lepMomentum = rt.TLorentzVector(lep[1], lep[2], lep[3], lep[0])
        nTrk = 0
        for h in event.EFlowTrack:
            if nTrk>=450: continue
            if h.PT<=0.5: continue
            q.SetPtEtaPhiM(h.PT, h.Eta, h.Phi, 0.)            
            if lepMomentum.DeltaR(q) > 0.0001:
                pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
                pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
                pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
                particles.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.PT, h.Eta, h.Phi, h.X, h.Y, h.Z, pfisoCh, pfisoGamma, pfisoNeu, 1, 0, 0, 0, 0, np.sign(h.PID)])
                nTrk += 1
        nPhoton = 0
        for h in event.EFlowPhoton:
            if nPhoton >= 150: continue
            if h.ET <= 1.: continue
            q.SetPtEtaPhiM(h.ET, h.Eta, h.Phi, 0.)
            pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
            pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
            pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
            particles.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.ET, h.Eta, h.Phi, 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 0, 1, 0, 0, 0])
            nPhoton += 1
        nNeu = 0
        for h in event.EFlowNeutralHadron:
            if nNeu >= 200: continue
            if h.ET <= 1.: continue
            q.SetPtEtaPhiM(h.ET, h.Eta, h.Phi, 0.)
            pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
            pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
            pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False) 
            particles.append([Nwritten, q.E(), q.Px(), q.Py(), q.Pz(), h.ET, h.Eta, h.Phi, 0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu, 0, 1, 0, 0, 0, 0])
            nNeu += 1
        for iTrk in range(nTrk, 450):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for iPhoton in range(nPhoton, 150):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for iNeu in range(nNeu, 200):
            particles.append([Nwritten, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # HIGH-LEVEL FEATURES
        myMET = event.MissingET[0]
        MET = myMET.MET
        phiMET = myMET.Phi
        MT = 2.*MET*lepMomentum.Pt()*(1-rt.TMath.Cos(lepMomentum.Phi()-phiMET))
        # (b)jet multiplicity
        HT = 0
        nJets = 0
        nBjets = 0
        for jet in event.Jet:
            if jet.PT > 40 and abs(jet.Eta)<2.4: 
                nJets +=1
                HT += jet.PT
                if jet.BTag >0: nBjets += 1
        LepPt = lep[5]
        LepEta = lep[6]
        LepPhi = lep[7]
        LepIsoCh = lep[11]
        LepIsoGamma = lep[12]
        LepIsoNeu = lep[13]
        LepCharge = lep[19]
        LepIsEle = lep[17]
        HLF.append([Nwritten, HT, MET, phiMET, MT, nJets, nBjets, LepPt, LepEta, LepPhi, LepIsoCh, LepIsoGamma, LepIsoNeu, LepCharge, LepIsEle])
        Nevt += 1
        Nwritten += 1
    ##### NOW SAVE INTO H5
    #HLF = HLF[15:]
    #nRows = int(HLF.shape[0]/15)
    #HLF = HLF.reshape((nRows,15))
    #HLFpandas = pd.DataFrame({'EvtId':HLF[:,0],'HT':HLF[:,1], 'MET':HLF[:,2], 'PhiMET':HLF[:,3], 'MT':HLF[:,4], 'nJets':HLF[:,5], 'bJets':HLF[:,6],\
    #                              'LepPt':HLF[:,7],'LepEta':HLF[:,8],'LepPhi':HLF[:,9],'LepIsoCh':HLF[:,10],'LepIsoGamma':HLF[:,11],'LepIsoNeu':HLF[:,12],\
    #                             'LepCharge':HLF[:,13],'LepIsEle':HLF[:,14]})
    #HLFpandas.to_hdf(filename.replace(".root",".h5"),'HLF')
    if len(HLF) !=0:
        fileOUT = filename.replace(".root",".h5")
        if minEvt != 0 or maxEvt != 10000:
            fileOUT = filename.replace(".root","_%i_TO_%i.h5" %(minEvt,maxEvt))
        f = h5py.File(fileOUT, "w")
        f.create_dataset('HLF', data=np.asarray(HLF), compression='gzip')
        f.create_dataset('HLF_Names', data=np.array(['EvtId','HT', 'MET', 'PhiMET', 'MT', 'nJets', 'bJets', 'LepPt', 'LepEta', 'LepPhi', \
                                                    'LepIsoCh', 'LepIsoGamma', 'LepIsoNeu', 'LepCharge', 'LepIsEle']), compression='gzip')
        pArray = np.asarray(particles).reshape((Nwritten,(200+150+450+1), 20))
        f.create_dataset('Particles', data=pArray, compression='gzip')
        f.create_dataset('Particles_Names', data=np.array(['EvtId', 'Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 'vtxX', 'vtxY', 'vtxZ', \
                                                             'ChPFIso', 'GammaPFIso', 'NeuPFIso', 'isChHad', 'isNeuHad', 'isGamma', 'isEle', 'isMu', 'Charge']),\
                             compression='gzip')
        f.close()

    #PARTICLESPanda = pd.DataFrame({'EvtId':particles[:,0], \
    #                               'Energy': particles[:,1], \
    #                               'Px': particles[:,2], \
    #                               'Py': particles[:,3], \
    #                               'Pz': particles[:,4], \
    #                               'Pt': particles[:,5], \
    #                               'Eta': particles[:,6], \
    #                               'Phi': particles[:,7], \
    #                               'vtxX': particles[:,8], \
    #                               'vtxY': particles[:,9], \
    #                               'vtxZ': particles[:,10], \
    #                               'ChPFIso': particles[:,11], \
    #                               'GammaPFIso': particles[:,12], \
    #                               'NeuPFIso': particles[:,13], \
    #                               'isChHad': particles[:,14], \
    #                               'isNeuHad': particles[:,15], \
    #                               'isGamma': particles[:,16], \
    #                               'isEle': particles[:,17], \
    #                               'isMu': particles[:,18], \
    #                               'Charge': particles[:,19]})
    #PARTICLESPanda.to_hdf(filename.replace(".root",".h5"),'Particles')
    #with open(filename.replace(".root",".csv"), "wb") as f:
    #    writer = csv.writer(f)
    #    writer.writerows(events)

if __name__ == "__main__":
    import sys
    print sys.argv[1]
    inFile = rt.TFile.Open(sys.argv[1])    
    tree = inFile.Get("Delphes")
    if len(sys.argv)==4:
        minIn = max(0, int(sys.argv[2]))
        maxEvt = min(tree.GetEntries(), int(sys.argv[3]))
    inFile.Close()
    if len(sys.argv)==4:
        Convert(sys.argv[1], minIn, maxEvt)
    else:
        Convert(sys.argv[1], 0, 10000)
