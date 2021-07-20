from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy as np
import uproot3 as u3
import awkward as ak

GLOBAL_PREFIX = ""

def uproot_root2array(fname, treename, stop=None, branches=None):
    dtypes = np.dtype( [(b, np.dtype("O")) for b in branches] )
    if isinstance(fname, list):
        fname = fname[0]
    tree = u3.open(fname)[treename]

    new_arr = np.empty( len(tree[branches[0]].array()), dtype=dtypes)

    for branch in branches:
        new_arr[branch] = np.array( ak.to_list( tree[branch].array() ), dtype="O") 

    return new_arr

def uproot_tree_to_numpy(fname, MeanNormTuple,inbranches_listlist, nMaxslist, nevents, treename="ttree", stop=None, branches=None):

    # array = uproot_root2array(fname, treename, stop=stop, branches=branches)

    # Read in total number of events
    totallengthperjet = 0
    for i in range(len(nMaxslist)):
        if nMaxslist[i]>=0:
            totallengthperjet+=len(inbranches_listlist[i])*nMaxslist[i]
        else:
            totallengthperjet+=len(inbranches_listlist[i]) #flat branch
    # branches = [ak.fill_none(ak.pad_none(tree[barr, target=feature_length), 0.) for feature_length, arr in zip( nMaxslist, inbranches_listlist)]
    tree = u3.open(fname)[treename]
    branches = [ak.fill_none(ak.pad_none( tree[branch_name].array(), target=feature_length, axis=-1, clip=True if feature_length > 1 else False), 0.) for feature_length, branch_list in zip( nMaxslist, inbranches_listlist) for branch_name in branch_list]

    branchnames = [n for names in inbranches_listlist for n in names]
    feature_lenghts = [f for branches, f in zip(inbranches_listlist, nMaxslist) for _ in branches]
    means = [m[0] for branches, m in zip(inbranches_listlist, MeanNormTuple) for _ in branches]
    norms = [m[1] for branches, m in zip(inbranches_listlist, MeanNormTuple) for _ in branches]
    print("Debugigng means and norms")
    print(means)
    print(norms)

    print(branchnames)
    branches_numpy = []
    for br, brname, fl, mean, norm in zip(branches, branchnames, feature_lenghts, means, norms):
        print("DBG {}".format(brname))
        print(br)
        print("Length: {}".format(len(br)))
        if brname == "TagVarCSV_trackJetDistVal":
            print("BONUS DEBUG!")
            print("Min: {}, Max: {}".format( ak.min( ak.count(br, axis=-1)),ak.max( ak.count(br, axis=-1))))
        if fl > 1:
            # branches_numpy.append( (ak.to_numpy( br ) - mean) / norm)
            branches_numpy.append( (ak.to_numpy( br ) - 0.) / 1.)
        elif fl == 1:
            # branches_numpy.append( (np.expand_dims( ak.to_numpy( br ), axis=-1) - mean)/norm  )
            branches_numpy.append( (np.expand_dims( ak.to_numpy( br ), axis=-1) - 0.)/1.  )
    print("FINISHED THIS LOOP, YOU ARE PERFECT! :) ")
    
    numpyarray = np.concatenate(branches_numpy, axis=-1)
    print("\n"*5)
    print("Some metrics about this numpy array")
    print( np.mean(numpyarray, axis=0))
    print( np.std(numpyarray, axis=0))
    print("Normalize array")
    numpyarray = (numpyarray - np.mean(numpyarray, axis=0) )/ np.std( numpyarray, axis=0)
    print("Some metrics about this numpy array")
    print( np.mean(numpyarray, axis=0))
    print( np.std(numpyarray, axis=0))
    return numpyarray



def uproot_MeanNormZeroPad(Filename_in,MeanNormTuple,inbranches_listlist, nMaxslist,nevents):
    # savely copy lists (pass by value)
    import copy
    inbranches_listlist=copy.deepcopy(inbranches_listlist)
    nMaxslist=copy.deepcopy(nMaxslist)
    
    # Read in total number of events
    totallengthperjet = 0
    for i in range(len(nMaxslist)):
        if nMaxslist[i]>=0:
            totallengthperjet+=len(inbranches_listlist[i])*nMaxslist[i]
        else:
            totallengthperjet+=len(inbranches_listlist[i]) #flat branch

    print("Total event-length per jet: {}".format(totallengthperjet))
    
    #shape could be more generic here... but must be passed to c module then
    array = numpy.zeros((nevents,totallengthperjet) , dtype='float32')

    # filling mean and normlist
    normslist=[]
    meanslist=[]
    for inbranches in inbranches_listlist:
        means=[]
        norms=[]
        for b in inbranches:
            if MeanNormTuple is None:
                means.append(0)
                norms.append(1)
            else:
                means.append(MeanNormTuple[b][0])
                norms.append(MeanNormTuple[b][1])
        meanslist.append(means)
        normslist.append(norms)

    # now start filling the array


def map_prefix(elements):
    if isinstance(elements, list):
        return list(map( lambda x: GLOBAL_PREFIX + x, elements))
    elif isinstance(elements, tuple):
        return tuple(map( lambda x: GLOBAL_PREFIX + x, elements))
    elif isinstance(elements, (str)):
        return GLOBAL_PREFIX + elements
    elif isinstance(elements, bytes):
        return GLOBAL_PREFIX + elements.decode("utf-8")
    else:
        print("Error, you gave >>{}<< which is unknown".format(elements))
        raise NotImplementedError

class TrainData_DF(TrainData):
    def __init__(self):

        TrainData.__init__(self)

        self.truth_branches = map_prefix(['Jet_isB','Jet_isBB','Jet_isGBB','Jet_isLeptonicB','Jet_isLeptonicB_C','Jet_isC','Jet_isGCC','Jet_isCC','Jet_isUD','Jet_isS','Jet_isG'])
        self.undefTruth=['Jet_isUndefined']
        self.weightbranchX=map_prefix('Jet_pt')
        self.weightbranchY=map_prefix('Jet_eta')
        self.remove = True
        self.referenceclass=map_prefix('Jet_isB')  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = map_prefix(['cat_B','cat_C','cat_UDS','cat_G']) #Reduced classes (flat only)
        self.truth_red_fusion = map_prefix([('Jet_isB','Jet_isBB','Jet_isGBB','Jet_isLeptonicB','Jet_isLeptonicB_C'),('Jet_isC','Jet_isGCC','Jet_isCC'),('Jet_isUD','Jet_isS'),('Jet_isG')]) #Indicate here how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = [1.00,1.00,2.50,5.00]  #Ratio between our reduced classes (flat only)
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        self.global_branches = map_prefix(['Jet_pt', 'Jet_eta',
                                'nCpfcand','nNpfcand',
                                'nsv','npv',
                                'TagVarCSV_trackSumJetEtRatio',
                                'TagVarCSV_trackSumJetDeltaR',
                                'TagVarCSV_vertexCategory',
                                'TagVarCSV_trackSip2dValAboveCharm',
                                'TagVarCSV_trackSip2dSigAboveCharm',
                                'TagVarCSV_trackSip3dValAboveCharm',
                                'TagVarCSV_trackSip3dSigAboveCharm',
                                'TagVarCSV_jetNTracks',
                                'TagVarCSV_jetNTracksEtaRel'])
                
        
        self.cpf_branches = map_prefix(['Cpfcan_BtagPf_trackEtaRel',
                             'Cpfcan_BtagPf_trackPtRel',
                             'Cpfcan_BtagPf_trackPPar',
                             'Cpfcan_BtagPf_trackDeltaR',
                             'Cpfcan_BtagPf_trackPParRatio',
                             'Cpfcan_BtagPf_trackSip2dVal',
                             'Cpfcan_BtagPf_trackSip2dSig',
                             'Cpfcan_BtagPf_trackSip3dVal',
                             'Cpfcan_BtagPf_trackSip3dSig',
                             'Cpfcan_BtagPf_trackJetDistVal',
                             'Cpfcan_ptrel',
                             'Cpfcan_drminsv',
                             'Cpfcan_VTX_ass',
                             'Cpfcan_puppiw',
                             'Cpfcan_chi2',
                             'Cpfcan_quality'])
        self.n_cpf = 25

        self.npf_branches = map_prefix(['Npfcan_ptrel','Npfcan_deltaR','Npfcan_isGamma','Npfcan_HadFrac','Npfcan_drminsv','Npfcan_puppiw'])
        self.n_npf = 25
        
        self.vtx_branches = map_prefix(['sv_pt','sv_deltaR',
                             'sv_mass',
                             'sv_ntracks',
                             'sv_chi2',
                             'sv_normchi2',
                             'sv_dxy',
                             'sv_dxysig',
                             'sv_d3d',
                             'sv_d3dsig',
                             'sv_costhetasvpv',
                             'sv_enratio',
        ])

        self.n_vtx = 4
        
        self.reduced_truth = map_prefix(['Jet_isB','Jet_isBB','Jet_isLeptonicB','Jet_isC','Jet_isUDS','Jet_isG'])

    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        weighter.class_weights = self.class_weights
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches, self.red_classes, 
                self.truth_red_fusion, method = self.referenceclass
            )
        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "ttree",
                    stop = None,
                    branches = branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
            #weighter.printHistos('/afs/cern.ch/user/a/ademoor/Flatten/') #If you need to print the 2D histo, choose your output dir
            return {'weigther':weighter}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        if not istraining:
            self.remove = False
        
        def reduceTruth(uproot_arrays):
            
            b = uproot_arrays[str.encode(map_prefix('Jet_isB'))]
            
            bb = uproot_arrays[str.encode(map_prefix('Jet_isBB'))]
            gbb = uproot_arrays[str.encode(map_prefix('Jet_isGBB'))]
            
            bl = uproot_arrays[map_prefix('Jet_isLeptonicB')]
            blc = uproot_arrays[map_prefix('Jet_isLeptonicB_C')]
            lepb = bl+blc
            
            c = uproot_arrays[map_prefix('Jet_isC')]
            cc = uproot_arrays[map_prefix('Jet_isCC')]
            gcc = uproot_arrays[map_prefix('Jet_isGCC')]
            
            ud = uproot_arrays[map_prefix('Jet_isUD')]
            s = uproot_arrays[map_prefix('Jet_isS')]
            uds = ud+s
            
            g = uproot_arrays[map_prefix('Jet_isG')]
            
            return np.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("ttree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.global_branches],
                                   [1],self.nsamples)

        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.cpf_branches,
                                   self.n_cpf,self.nsamples)

        x_npf = MeanNormZeroPadParticles(filename,None,
                                         self.npf_branches,
                                         self.n_npf,self.nsamples)

        x_vtx = MeanNormZeroPadParticles(filename,None,
                                         self.vtx_branches,
                                         self.n_vtx,self.nsamples)
        
        import uproot3 as uproot
        urfile = uproot.open(filename)["ttree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        x_cpf = x_cpf.astype(dtype='float32', order='C')
        x_npf = x_npf.astype(dtype='float32', order='C')
        x_vtx = x_vtx.astype(dtype='float32', order='C')
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "ttree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['Jet_isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_vtx=x_vtx[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')

        print('remove nans')
        x_global = np.where(np.isfinite(x_global) , x_global, 0)
        x_cpf = np.where(np.isfinite(x_cpf), x_cpf, 0)
        x_npf = np.where(np.isfinite(x_npf), x_npf, 0)
        x_vtx = np.where(np.isfinite(x_vtx), x_vtx, 0)

        return [x_global,x_cpf,x_npf,x_vtx], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isB, prob_isBB,prob_isLeptB, prob_isC,prob_isUDS,prob_isG,Jet_isB, Jet_isBB, isLeptB, Jet_isC,Jet_isUDS,Jet_isG,Jet_pt, Jet_eta')
        array2root(out, outfilename, 'tree')

class TrainData_DeepCSV(TrainData):
    def __init__(self):

        TrainData.__init__(self)

        self.description = "DeepCSV training datastructure"

        self.truth_branches = map_prefix(['Jet_isB','Jet_isBB','Jet_isGBB','Jet_isLeptonicB','Jet_isLeptonicB_C','Jet_isC','Jet_isGCC','Jet_isCC','Jet_isUD','Jet_isS','Jet_isG', ])
        self.undefTruth=['Jet_isUndefined']
        self.deepcsv_branches = map_prefix(['Jet_DeepFlavourBDisc',
                     'Jet_DeepFlavourCvsLDisc',
                     'Jet_DeepFlavourCvsBDisc',
                     'Jet_DeepFlavourB',
                     'Jet_DeepFlavourBB',
                     'Jet_DeepFlavourLEPB',
                     'Jet_DeepFlavourC',
                     'Jet_DeepFlavourUDS',
                     'Jet_DeepFlavourG',
                     'Jet_DeepCSVBDisc',
                     'Jet_DeepCSVBDiscN',
                     'Jet_DeepCSVCvsLDisc',
                     'Jet_DeepCSVCvsLDiscN',
                     'Jet_DeepCSVCvsBDisc',
                     'Jet_DeepCSVCvsBDiscN',
                     'Jet_DeepCSVb',
                     'Jet_DeepCSVc',
                     'Jet_DeepCSVl',
                     'Jet_DeepCSVbb',
                     'Jet_DeepCSVcc',
                     'Jet_DeepCSVbN',
                     'Jet_DeepCSVcN',
                     'Jet_DeepCSVlN',
                     'Jet_DeepCSVbbN',
                     'Jet_DeepCSVccN'])
        self.weightbranchX=map_prefix('Jet_pt')
        self.weightbranchY=map_prefix('Jet_eta')
        self.remove = True
        self.referenceclass=map_prefix('Jet_isB')  #Choose 'flatten' for flat or one of the truth branch for ref
        self.red_classes = map_prefix(['cat_B','cat_C','cat_UDS','cat_G']) #Reduced classes (flat only)
        self.truth_red_fusion = [map_prefix(('Jet_isB','Jet_isBB','Jet_isGBB','Jet_isLeptonicB','Jet_isLeptonicB_C')),map_prefix(('Jet_isC','Jet_isGCC','Jet_isCC')),map_prefix(('Jet_isUD','Jet_isS')),map_prefix(('Jet_isG'))] #Indicate here how you are making the fusion of your truth branches to the reduced classes for the flat reweighting
        self.class_weights = [1.00,1.00,2.50,5.00]  #Ratio between our reduced classes (flat only)
        self.weight_binX = np.array([
            10,25,30,35,40,45,50,60,75,100,
            125,150,175,200,250,300,400,500,
            600,2000],dtype=float)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
        )

        self.global_branches = map_prefix(['Jet_pt', 'Jet_eta',
                                'TagVarCSV_jetNSecondaryVertices', 
                                'TagVarCSV_trackSumJetEtRatio',
                                'TagVarCSV_trackSumJetDeltaR',
                                'TagVarCSV_vertexCategory',
                                'TagVarCSV_trackSip2dValAboveCharm',
                                'TagVarCSV_trackSip2dSigAboveCharm',
                                'TagVarCSV_trackSip3dValAboveCharm',
                                'TagVarCSV_trackSip3dSigAboveCharm',
                                'TagVarCSV_jetNTracks',
                                'TagVarCSV_jetNTracksEtaRel'])

        self.track_branches = map_prefix(['TagVarCSV_trackJetDistVal',
                              'TagVarCSV_trackPtRel', 
                              'TagVarCSV_trackDeltaR', 
                              'TagVarCSV_trackPtRatio', 
                              'TagVarCSV_trackSip3dSig', 
                              'TagVarCSV_trackSip2dSig', 
                              'TagVarCSV_trackDecayLenVal'])
        self.n_track = 6
        
        self.eta_rel_branches = map_prefix(['TagVarCSV_trackEtaRel'])
        self.n_eta_rel = 4

        self.vtx_branches = map_prefix(['TagVarCSV_vertexMass', 
                          'TagVarCSV_vertexNTracks', 
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexJetDeltaR',
                          'TagVarCSV_flightDistance2dVal', 
                          'TagVarCSV_flightDistance2dSig', 
                          'TagVarCSV_flightDistance3dVal', 
                          'TagVarCSV_flightDistance3dSig'])
        self.n_vtx = 1
                
        self.reduced_truth = map_prefix(['Jet_isB','Jet_isBB','Jet_isC','Jet_isUDSG'])

    def readTreeFromRootToTuple(self, filenames, limit=None, branches=None):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        
        can also read a list of files (e.g. to produce weights/removes from larger statistics
        (not fully tested, yet)
        '''
        
        if branches is None or len(branches) == 0:
            return np.array([],dtype='float32')
            
        #print(branches)
        #remove duplicates
        usebranches=list(set(branches))
        tmpbb=[]
        for b in usebranches:
            if len(b):
                tmpbb.append(b)
        usebranches=tmpbb
            
        import ROOT
        from root_numpy import tree2array, root2array
        if isinstance(filenames, list):
            for f in filenames:
                fileTimeOut(f,120)
            print('add files')
            print('debugging this')
            print("Branches:\n{}".format(usebranches))

            import uproot as ur
            import awkward as ak
            # this was substituted from the old root2array function
            nparray = uproot_root2array(
                filenames, 
                treename = "ttree", 
                stop = limit,
                branches = usebranches
                )
            print('done add files')
            return nparray
        else:    
            fileTimeOut(filenames,120) #give eos a minute to recover
            rfile = ROOT.TFile(filenames)
            tree = rfile.Get(self.treename)
            if not self.nsamples:
                self.nsamples=tree.GetEntries()
            nparray = tree2array(tree, stop=limit, branches=usebranches)
            return nparray
        
    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        weighter.class_weights = self.class_weights
        branches = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,self.weight_binY],
                self.weightbranchX,self.weightbranchY,
                self.truth_branches, self.red_classes, 
                self.truth_red_fusion, method = self.referenceclass
            )

        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = uproot_root2array(
                    fname,
                    treename = "ttree",
                    stop = None,
                    branches = branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                # from IPython import embed;embed()
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)

        print("calculate means")
        print("debugging this point here!")
        from DeepJetCore.preprocessing import meanNormProd
        nparray = self.readTreeFromRootToTuple(allsourcefiles,branches=self.vtx_branches+self.eta_rel_branches+self.track_branches+self.global_branches, limit=500000)
        print("read tree from sourcefiles")
        for a in (self.vtx_branches+self.eta_rel_branches+self.track_branches+self.global_branches):
            for b in range(len(nparray[a])):
                nparray[a][b] = np.where(np.logical_and(np.isfinite(nparray[a][b]),np.abs(nparray[a][b]) < 100000.0), nparray[a][b], 0)
        means = np.array([],dtype='float32')
        if len(nparray):
            means = meanNormProd(nparray)
        print("weigheter created")
        return {'weigther':weighter,'means':means}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw=stopwatch()
        swall=stopwatch()
        if not istraining:
            self.remove = False
                
        def reduceTruth(uproot_arrays):
            
            b = uproot_arrays[str.encode(map_prefix(b'Jet_isB'))]
            
            bb = uproot_arrays[str.encode(map_prefix(b'Jet_isBB'))]
            gbb = uproot_arrays[str.encode(map_prefix(b'Jet_isGBB'))]
            
            bl = uproot_arrays[str.encode(map_prefix(b'Jet_isLeptonicB'))]
            blc = uproot_arrays[str.encode(map_prefix(b'Jet_isLeptonicB_C'))]
            lepb = bl+blc
            
            c = uproot_arrays[str.encode(map_prefix(b'Jet_isC'))]
            cc = uproot_arrays[str.encode(map_prefix(b'Jet_isCC'))]
            gcc = uproot_arrays[str.encode(map_prefix(b'Jet_isGCC'))]
            
            ud = uproot_arrays[str.encode(map_prefix(b'Jet_isUD'))]
            s = uproot_arrays[str.encode(map_prefix(b'Jet_isS'))]
            uds = ud+s
            
            g = uproot_arrays[str.encode(map_prefix(b'Jet_isG'))]
            
            return np.vstack((b+lepb,bb+gbb,c+cc+gcc,uds+g)).transpose()
        
        print('reading '+filename)
        
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,600) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        # tree = rfile.Get("ttree")
        # self.nsamples = tree.GetEntries()  
        # from IPython import embed;embed()
        tree = u3.open(filename)["ttree"]
        self.nsamples = tree.numentries
        print("Nsamples: {}".format(self.nsamples))

        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        for obj in [filename,weighterobjects['means'],[self.global_branches,self.track_branches,self.eta_rel_branches,self.vtx_branches],[1,self.n_track,self.n_eta_rel,self.n_vtx],self.nsamples]:
            print("DEBUGGING:\t{}".format(type(obj)))
        print("DEBUGGING:\n\tPrinting MeanNormZeroPad arguments:")
        print("\t{}\n\t{}\n\t{}".format(filename, weighterobjects['means'],self.nsamples))
        print("reading in with new uproot+awkward function")
        nparr = uproot_tree_to_numpy(filename, weighterobjects['means'],
                                   [self.global_branches,self.track_branches,self.eta_rel_branches,self.vtx_branches],
                                   [1,self.n_track,self.n_eta_rel,self.n_vtx],self.nsamples, treename="ttree")
        print("succesfully created numpy array")
        x_global = nparr

        
        # x_global = MeanNormZeroPad(filename,weighterobjects['means'],
                                   # [self.global_branches,self.track_branches,self.eta_rel_branches,self.vtx_branches],
                                   # [1,self.n_track,self.n_eta_rel,self.n_vtx],self.nsamples)
                
        print("opening file with uproot")
        import uproot3 as uproot
        urfile = uproot.open(filename)["ttree"]
        truth_arrays = urfile.arrays(self.truth_branches)
        print("truth_branches:")
        print(self.truth_branches)
        print("truth_arrays:")
        print(truth_arrays)
        truth = reduceTruth(truth_arrays)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global = x_global.astype(dtype='float32', order='C')
        
        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = uproot_root2array(
                filename,
                treename = "ttree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            undef=for_remove['Jet_isUndefined']
            print("\nundef:")
            print(undef)
            print("undef dtype: ", undef.dtype)
            print()
            print(notremoves)
            notremoves -= np.array(undef, dtype=np.float32)
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.remove:
            print('remove')
            x_global=x_global[notremoves > 0]
            truth=truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        
        print('remove nans')
        x_global = np.where(np.logical_and(np.isfinite(x_global), (np.abs(x_global) < 100000.0)), x_global, 0)
        return [x_global], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        spectator_branches = ['Jet_pt','Jet_eta']
        spectator_branches += self.deepcsv_branches
        from root_numpy import array2root
        # if inputfile[-5:] == 'djctd':
            # print("storing normed pt and eta... run on root files if you want something else for now")
            # spectators = features[0][:,0:2].transpose()
        # else:
        import uproot3 as uproot
        print(inputfile)
        urfile = uproot.open(inputfile)["ttree"]
        spectator_arrays = urfile.arrays(spectator_branches)
        print(spectator_arrays)
        spectators = [spectator_arrays[a.encode()] for a in spectator_branches]
        
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), spectators) ),
                                         names='prob_isB, prob_isBB, prob_isC,prob_isUDSG,Jet_isB, Jet_isBB, Jet_isC,Jet_isUDSG,Jet_pt,Jet_eta,'+",".join(map(str, self.deepcsv_branches)))
        array2root(out, outfilename, 'tree')
