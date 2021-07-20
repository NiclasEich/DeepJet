print("start import")
import os
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import InterpolatedUnivariateSpline
from pdb import set_trace
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import root_numpy
import ROOT
from ROOT import TCanvas, TGraph, TGraphAsymmErrors, TH2F, TH1F
print("finish import")
from root_numpy import fill_hist


def spit_out_roc(disc,truth_array,selection_array):

    newx = np.logspace(-3.5, 0, 100)
    tprs = pd.DataFrame()
    truth = truth_array[selection_array]*1
    disc = disc[selection_array]
    tmp_fpr, tmp_tpr, _ = roc_curve(truth, disc)
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    tprs = spline(newx)
    return tprs, newx



pred = []
isDeepJet = False
if isDeepJet:
    listbranch = ['prob_isB', 'prob_isBB','prob_isLeptB', 'prob_isC','prob_isUDS','prob_isG','Jet_isB', 'Jet_isBB', 'Jet_isLeptB', 'Jet_isC','Jet_isUDS','Jet_isG','Jet_pt', 'Jet_eta'] + ['Jet_DeepFlavourBDisc',
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
                     'Jet_DeepCSVccN']
else:
    listbranch = ['prob_isB', 'prob_isBB', 'prob_isC','prob_isUDSG','Jet_isB', 'Jet_isBB', 'Jet_isC','Jet_isUDSG','Jet_pt', 'Jet_eta'] + ['Jet_DeepFlavourBDisc',
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
                     'Jet_DeepCSVccN']

# dirz = '/data/ml/ebols/DeepCSV_PredictionsTest/'
# dirz = 'data/ml/ebols/DeepCSV_PredictionsTest/'
# dirz = '/afs/cern.ch/work/n/neich/private/BTV-HLT-training-tools/training_output/_pred/'
dirz = os.path.join( os.getenv("TrainingOutput"), os.getenv("TrainingVersion")+"_pred/")
truthfile = open( os.path.join(dirz, 'outfiles.txt'), 'r')
print("opened text file")
rfile1 = ROOT.TChain("tree")
count = 0

for line in truthfile:
    count += 1
    if len(line) < 1: continue
    file1name=str(dirz+line.split('\n')[0])
    rfile1.Add(file1name)

print("added files")
df = root_numpy.tree2array(rfile1, branches = listbranch)
print("converted to root")

if isDeepJet:
    b_jets = df['Jet_isB']+df['Jet_isBB']+df['Jet_isLeptB']
    disc = df['prob_isB']+df['prob_isBB']+df['prob_isLeptB']
    summed_truth = df['Jet_isB']+df['Jet_isBB']+df['Jet_isLeptB']+df['Jet_isC']+df['Jet_isUDS']+df['Jet_isG']
    veto_c = (df['Jet_isC'] != 1) & ( df['Jet_pt'] > 30) & (summed_truth != 0)
    veto_udsg = (df['Jet_isUDS'] != 1) & (df['Jet_isG'] != 1) & ( df['Jet_pt'] > 30) & (summed_truth != 0)
else:
    b_jets = df['Jet_isB']+df['Jet_isBB']
    disc = df['prob_isB']+df['prob_isBB']
    summed_truth = df['Jet_isB']+df['Jet_isBB']+df['Jet_isC']+df['Jet_isUDSG']
    veto_c = (df['Jet_isC'] != 1) & ( df['Jet_pt'] > 30) & (summed_truth != 0)
    veto_udsg = (df['Jet_isUDSG'] != 1) & ( df['Jet_pt'] > 30) & (summed_truth != 0)

    disc_DeepCSV = df['Jet_DeepCSVb']+df['Jet_DeepCSVbb']
    # veto_c_DeepCSV = (df['Jet_DeepCSVc'] != 1) & ( df['Jet_pt'] > 30) & (summed_truth != 0)
    # veto_udsg_DeepCSV = (df['Jet_DeepCSVl'] != 1) & ( df['Jet_pt'] > 30) & (summed_truth != 0)


f = ROOT.TFile(os.path.join(dirz, "ROCS_DeepCSV.root"), "recreate")

x1, y1 = spit_out_roc(disc,b_jets,veto_c)
x2, y2 = spit_out_roc(disc,b_jets,veto_udsg)

x3, y3 = spit_out_roc(disc_DeepCSV,b_jets,veto_c)
x4, y4 = spit_out_roc(disc_DeepCSV,b_jets,veto_udsg)

gr1 = TGraph( 100, x1, y1 )
gr1.SetName("roccurve_0")
gr2 = TGraph( 100, x2, y2 )
gr2.SetName("roccurve_1")
gr3 = TGraph( 100, x3, y3 )
gr3.SetName("roccurve_2")
gr4 = TGraph( 100, x4, y4 )
gr4.SetName("roccurve_3")
gr1.Write()
gr2.Write()
gr3.Write()
gr4.Write()
f.Write()
