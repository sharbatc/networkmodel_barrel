'''
    Project Disinhibition
    Containing whole parameters of network and neurons
    Started : 30 March, 2015 by Hesam SETAREH
    Last modified : 18 Jun, 2018 by Sharbat
'''

import preprocessing
import numpy as np
import pandas as pd


## tested params, might have to be changed##
param_set_exc = 0 #1 for Carl-Hesam's hub neurons, 5 for Hesam's newest
param_set_nfs = 0 
param_set_fs  = 0
## IMP : CHANGED PARAMS ARE NOT AT 0 
## tested params, might have to be changed ##

## loading the data
exc_df = pd.read_csv('data/exc.txt',header = None)
nfs_df = pd.read_csv('data/nfs.txt',header = None)
fs_df = pd.read_csv('data/fs.txt',header = None)

## naming the columns
columns_name_list = ['C (nF)','gL (usiemens)','v_rest (mV)','v_reset (mV)','tau_refr (ms)',\
             'v_thresh (mV)','del_v (mV)', 'amp_w1 (nA)','tau_w1 (ms)','amp_w2 (nA)',\
             'tau_w2 (ms)','amp_vt1 (mV)','tau_vt1 (ms)','amp_vt2 (mV)','tau_vt2 (ms)']

exc_df.columns = columns_name_list
nfs_df.columns = columns_name_list
fs_df.columns = columns_name_list

exc_param = exc_df.iloc[param_set_exc,:]
nfs_param = nfs_df.iloc[param_set_nfs,:]
fs_param = fs_df.iloc[param_set_fs,:]

###### cell numbers (population size)

# New numbers, exc from Lefort, other numbers from Taro experiment only in L2/3 and L4(real and experimentally extracted numbers)

# Other numbers are from Lefort's total imhibitory neurons, divided according to Rudy 2013 ratios

# Important : new numbers are fs/nfs/exc

# numbers 
#L2_size = {'exc':546, 'sst':22, 'vip':32, 'pv':34}
#L3_size = {'exc':1145, 'sst':45, 'vip':63,'pv':67}

L23_size = {'exc':1691, 'nfs':162, 'fs': 101}

L4_size  = {'exc':1656, 'nfs':28, 'fs': 87} # added the neurons of sst and vip to make nfs neurons

#L5A_size = {'exc' : 454, 'sst' : 40, 'vip' : 7, 'pv' : 40} 
#L5B_size = {'exc' : 641, 'sst' : 60, 'vip' : 7, 'pv' : 60} 
#L6_size = {'exc' : 1288, 'sst' : 50, 'vip' : 20, 'pv' : 50}


size = {'L23':L23_size, 'L4':L4_size}
# size = {'L2':L2_size, 'L3':L3_size, 'L4':L4_size, 'L5A':L5A_size, 'L5B':L5B_size, 'L6':L6_size}


###### neural parameters

## pv is fs seamlessly 
## sst and vip are for the moment taken to be nfs
## note two, take only nfs for the moment, so, only use pv

L23_exc= {}
L23_nfs= {}
L23_fs = {}


L23_exc['R'] = 1.0/exc_param['gL (usiemens)']
L23_fs['R'] = 1.0/fs_param['gL (usiemens)']
L23_nfs['R'] = 1.0/nfs_param['gL (usiemens)']

L23_exc['tau_m'] = exc_param['C (nF)']/exc_param['gL (usiemens)']
L23_fs['tau_m'] = fs_param['C (nF)']/fs_param['gL (usiemens)']
L23_nfs['tau_m'] = nfs_param['C (nF)']/nfs_param['gL (usiemens)']

# L2_exc['R'] = 1.0/exc_param[1]
# L2_pv['R'] = 1.0/fs_param[1]
# L2_sst['R'] = 1.0/nfs_param[1]
# L2_vip['R'] = 1.0/nfs_param[1]

# L3_exc['R'] = 1.0/exc_param[1]
# L3_pv['R'] = 1.0/fs_param[1]
# L3_sst['R'] = 1.0/nfs_param[1]
# L3_vip['R'] = 1.0/nfs_param[1]

# L2_exc['tau_m'] = exc_param[0]/exc_param[1]
# L2_vip['tau_m'] = nfs_param[0]/nfs_param[1]
# L2_sst['tau_m'] = nfs_param[0]/nfs_param[1]
# L2_pv['tau_m'] = fs_param[0]/fs_param[1]

# L3_exc['tau_m'] = exc_param[0]/exc_param[1]
# L3_vip['tau_m'] = nfs_param[0]/nfs_param[1]
# L3_sst['tau_m'] = nfs_param[0]/nfs_param[1]
# L3_pv['tau_m'] = fs_param[0]/fs_param[1]

L4_exc = {}
L4_fs = {}
L4_nfs= {}

## L4 vip is not modelled 
L4_exc['R'] = 1.0/exc_param['gL (usiemens)']
L4_fs['R'] = 1.0/fs_param['gL (usiemens)']
L4_nfs['R'] = 1.0/nfs_param['gL (usiemens)']

L4_exc['tau_m'] = exc_param['C (nF)']/exc_param['gL (usiemens)']
L4_fs['tau_m'] = fs_param['C (nF)']/fs_param['gL (usiemens)']
L4_nfs['tau_m'] = nfs_param['C (nF)']/nfs_param['gL (usiemens)']


# ## L5A
# L5A_exc['R'] = 1.0/exc_param[1]
# L5A_pv['R'] = 1.0/fs_param[1]
# L5A_sst['R'] = 1.0/nfs_param[1]

# L5A_exc['tau_m'] = exc_param[0]/exc_param[1]
# L5A_sst['tau_m'] = nfs_param[0]/nfs_param[1]
# L5A_pv['tau_m'] = fs_param[0]/fs_param[1]


# ## L5B
# L5B_exc['R'] = 1.0/exc_param[1]
# L5B_pv['R'] = 1.0/fs_param[1]
# L5B_sst['R'] = 1.0/nfs_param[1]

# L5B_exc['tau_m'] = exc_param[0]/exc_param[1]
# L5B_sst['tau_m'] = nfs_param[0]/nfs_param[1]
# L5B_pv['tau_m'] = fs_param[0]/fs_param[1]

# ## L6
# L6_exc['R'] = 1.0/exc_param[1]
# L6_pv['R'] = 1.0/fs_param[1]
# L6_sst['R'] = 1.0/nfs_param[1]

# L6_exc['tau_m'] = exc_param[0]/exc_param[1]
# L6_sst['tau_m'] = nfs_param[0]/nfs_param[1]
# L6_pv['tau_m'] = fs_param[0]/fs_param[1]



## All params
# L2_neural = {'exc':L2_exc, 'sst':L2_sst, 'vip':L2_vip, 'pv':L2_pv}
# L3_neural = {'exc':L3_exc, 'sst':L3_sst, 'vip':L3_vip, 'pv':L3_pv}
L23_neural = {'exc':L23_exc, 'nfs':L23_nfs, 'fs':L23_fs}
L4_neural  = {'exc':L4_exc, 'nfs':L4_nfs, 'fs':L4_fs}
# L5A_neural = {'exc':L5A_exc, 'sst':L5A_exc, 'pv':L5A_pv}
# L5B_neural = {'exc':L5B_exc, 'sst':L5B_exc, 'pv':L5B_pv}
# L6_neural = {'exc':L6_exc, 'sst':L6_exc, 'pv':L6_pv}

neural_param = {'L23':L23_neural, 'L4':L4_neural}

# neural_param = {'L2':L2_neural, 'L3':L3_neural, 'L4':L4_neural, 'L5A':L5A_neural, 'L5B':L5B_neural, 'L6':L6_neural}

###################### connection parameters

conn_param = {'L23_L23':None, 'L4_L4':None, 'L4_L23':None}

#### inside L2/3
conn_param['L23_L23'] = {}


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 26.2, neural_param['L23']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L23_exc to L23_exc, avermann 2011
conn_param['L23_L23']['exc_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.168} 

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.82, 13.7, neural_param['L23']['fs']['tau_m'], neural_param['L23']['fs']['R']) # L23_exc to L23_fs, avermann 2011
conn_param['L23_L23']['exc_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.575}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.52, 43.1, neural_param['L23']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L23_fs to L23_exc, avermann 2011
conn_param['L23_L23']['fs_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.60}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.56, 15.8, neural_param['L23']['fs']['tau_m'], neural_param['L23']['fs']['R']) # L23_pv to L23_pv, avermann 2011
conn_param['L23_L23']['fs_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.55}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.244, 17.9, neural_param['L23']['nfs']['tau_m'], neural_param['L23']['nfs']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['exc_nfs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.244}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 56.2, neural_param['L23']['exc']['tau_m'], neural_param['L23']['exc']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['nfs_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.465}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L23']['nfs']['tau_m'], neural_param['L23']['nfs']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['nfs_nfs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L23']['fs']['tau_m'], neural_param['L23']['fs']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['nfs_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 33.3, neural_param['L23']['nfs']['tau_m'], neural_param['L23']['nfs']['R']) # avermann 2011, wrong half-width
conn_param['L23_L23']['fs_nfs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.379}

#### inside L4
conn_param['L4_L4'] = {}


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.95, 50.0, neural_param['L4']['exc']['tau_m'], neural_param['L4']['exc']['R']) # L4_exc to L4_exc, lefort 2009
conn_param['L4_L4']['exc_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.243} 

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.82, 13.7, neural_param['L4']['fs']['tau_m'], neural_param['L4']['fs']['R']) # L4_exc to L4_pv, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['exc_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.69}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.52, 43.1, neural_param['L4']['exc']['tau_m'], neural_param['L4']['exc']['R']) # L4_pv to L4_exc, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['fs_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.67}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.56, 15.8, neural_param['L4']['fs']['tau_m'], neural_param['L4']['fs']['R']) # L4_pv to L4_pv, all copy from L2/3
conn_param['L4_L4']['fs_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.55}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.244, 17.9, neural_param['L4']['nfs']['tau_m'], neural_param['L4']['nfs']['R']) # nfs paramteres of avermann 2011 instead of sst, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['exc_nfs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.40}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 56.2, neural_param['L4']['exc']['tau_m'], neural_param['L4']['exc']['R']) # nfs paramteres of avermann 2011 instead of sst, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['nfs_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.29}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L4']['nfs']['tau_m'], neural_param['L4']['nfs']['R']) # nfs paramteres of avermann 2011 instead of sst, all copy from L2/3
conn_param['L4_L4']['nfs_nfs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 20.0, neural_param['L4']['fs']['tau_m'], neural_param['L4']['fs']['R']) # L4_sst to L4_pv, actually vip to pv of L23, p adopfted from Taro's presentation
conn_param['L4_L4']['nfs_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.63}

## fs to nfs? (Avermann?)

#### L4 to L2/3
conn_param['L4_L23'] = {}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.6, 31.0, neural_param['L4']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L4_exc to L23_exc, lefort 2009
conn_param['L4_L23']['exc_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.13} 


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.50, 31.0, neural_param['L23']['fs']['tau_m'], neural_param['L23']['fs']['R']) # L4_exc to L23_pv, lefort 2009
conn_param['L4_L23']['exc_fs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.2} 

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.50, 31.0, neural_param['L23']['nfs']['tau_m'], neural_param['L23']['nfs']['R']) # L4_exc to L23_nfs, lefort 2009
conn_param['L4_L23']['exc_nfs'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.2} 

