'''
    Project Disinhibition
    Containing whole parameters of network and neurons
    Started : 30 March, 2015 by Hesam SETAREH
    Last modified : 11 Jun, 2018 by Sharbat
'''

import preprocessing
import numpy


## tested params, might have to be changed##
param_set_exc = 3
param_set_nfs = 2 
param_set_fs  = 1 
## tested params, might have to be changed ##

exc_param = numpy.loadtxt('data/exc.txt', delimiter = ',')[param_set_exc]
nfs_param = numpy.loadtxt('data/nfs.txt', delimiter = ',')[param_set_nfs]
fs_param  = numpy.loadtxt('data/fs.txt' , delimiter = ',')[param_set_fs]


###### cell numbers (population size)

# New numbers, exc from Lefort, other numbers from Taro experiment only in L2/3 and L4(real and experimentally extracted numbers)

# Other numbers are from Lefort's total imhibitory neurons, divided according to Rudy 2013 ratios

# numbers 
L2_size = {'exc':546, 'sst':22, 'vip':32, 'pv':34}
L3_size = {'exc':1145, 'sst':45, 'vip':63,'pv':67}
L4_size  = {'exc':1656, 'sst':21, 'vip':7, 'pv': 87} # hesam did not implement vip neurons for L4
L5A_size = {'exc' : 454, 'sst' : 40, 'vip' : 7, 'pv' : 40} 
L5B_size = {'exc' : 641, 'sst' : 60, 'vip' : 7, 'pv' : 60} 
L6_size = {'exc' : 1288, 'sst' : 50, 'vip' : 20, 'pv' : 50}


size = {'L2':L2_size, 'L3':L3_size, 'L4':L4_size, 'L5A':L5A_size, 'L5B':L5B_size, 'L6':L6_size}


###### neural parameters

## pv is fs seamlessly 
## sst and vip are for the moment taken to be nfs

L2_exc= {}
L2_vip= {}
L2_pv = {}
L2_sst= {}

L3_exc= {}
L3_vip= {}
L3_pv = {}
L3_sst= {}

L4_exc= {}
L4_vip= {}
L4_pv = {}
L4_sst= {}

L5A_exc= {}
L5A_vip= {}
L5A_pv = {}
L5A_sst= {}

L5B_exc= {}
L5B_vip= {}
L5B_pv = {}
L5B_sst= {}

L6_exc= {}
L6_vip= {}
L6_pv = {}
L6_sst= {}

L2_exc['R'] = 1.0/exc_param[1]
L2_pv['R'] = 1.0/fs_param[1]
L2_sst['R'] = 1.0/nfs_param[1]
L2_vip['R'] = 1.0/nfs_param[1]

L3_exc['R'] = 1.0/exc_param[1]
L3_pv['R'] = 1.0/fs_param[1]
L3_sst['R'] = 1.0/nfs_param[1]
L3_vip['R'] = 1.0/nfs_param[1]

L2_exc['tau_m'] = exc_param[0]/exc_param[1]
L2_vip['tau_m'] = nfs_param[0]/nfs_param[1]
L2_sst['tau_m'] = nfs_param[0]/nfs_param[1]
L2_pv['tau_m'] = fs_param[0]/fs_param[1]

L3_exc['tau_m'] = exc_param[0]/exc_param[1]
L3_vip['tau_m'] = nfs_param[0]/nfs_param[1]
L3_sst['tau_m'] = nfs_param[0]/nfs_param[1]
L3_pv['tau_m'] = fs_param[0]/fs_param[1]

## L4 vip is not modelled 
L4_exc['R'] = 1.0/exc_param[1]
L4_pv['R'] = 1.0/fs_param[1]
L4_sst['R'] = 1.0/nfs_param[1]

L4_exc['tau_m'] = exc_param[0]/exc_param[1]
L4_sst['tau_m'] = nfs_param[0]/nfs_param[1]
L4_pv['tau_m'] = fs_param[0]/fs_param[1]

## L5A
L5A_exc['R'] = 1.0/exc_param[1]
L5A_pv['R'] = 1.0/fs_param[1]
L5A_sst['R'] = 1.0/nfs_param[1]

L5A_exc['tau_m'] = exc_param[0]/exc_param[1]
L5A_sst['tau_m'] = nfs_param[0]/nfs_param[1]
L5A_pv['tau_m'] = fs_param[0]/fs_param[1]


## L5B
L5B_exc['R'] = 1.0/exc_param[1]
L5B_pv['R'] = 1.0/fs_param[1]
L5B_sst['R'] = 1.0/nfs_param[1]

L5B_exc['tau_m'] = exc_param[0]/exc_param[1]
L5B_sst['tau_m'] = nfs_param[0]/nfs_param[1]
L5B_pv['tau_m'] = fs_param[0]/fs_param[1]

## L6
L6_exc['R'] = 1.0/exc_param[1]
L6_pv['R'] = 1.0/fs_param[1]
L6_sst['R'] = 1.0/nfs_param[1]

L6_exc['tau_m'] = exc_param[0]/exc_param[1]
L6_sst['tau_m'] = nfs_param[0]/nfs_param[1]
L6_pv['tau_m'] = fs_param[0]/fs_param[1]

## All params
L2_neural = {'exc':L2_exc, 'sst':L2_sst, 'vip':L2_vip, 'pv':L2_pv}
L3_neural = {'exc':L3_exc, 'sst':L3_sst, 'vip':L3_vip, 'pv':L3_pv}
L4_neural  = {'exc':L4_exc, 'sst':L4_sst, 'pv':L4_pv}
L5A_neural = {'exc':L5A_exc, 'sst':L5A_exc, 'pv':L5A_pv}
L5B_neural = {'exc':L5B_exc, 'sst':L5B_exc, 'pv':L5B_pv}
L6_neural = {'exc':L6_exc, 'sst':L6_exc, 'pv':L6_pv}


neural_param = {'L2':L2_neural, 'L3':L3_neural, 'L4':L4_neural, 'L5A':L5A_neural, 'L5B':L5B_neural, 'L6':L6_neural}

###################### connection parameters

conn_param = {'L23_L23':None, 'L4_L4':None, 'L4_L23':None}

#### inside L2/3
conn_param['L23_L23'] = {}


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 26.2, neural_param['L2']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L23_exc to L23_exc, avermann 2011
conn_param['L23_L23']['exc_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.168} 

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.82, 13.7, neural_param['L23']['pv']['tau_m'], neural_param['L23']['pv']['R']) # L23_exc to L23_pv, avermann 2011
conn_param['L23_L23']['exc_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.575}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.52, 43.1, neural_param['L23']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L23_pv to L23_exc, avermann 2011
conn_param['L23_L23']['pv_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.60}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.56, 15.8, neural_param['L23']['pv']['tau_m'], neural_param['L23']['pv']['R']) # L23_pv to L23_pv, avermann 2011
conn_param['L23_L23']['pv_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.55}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.244, 17.9, neural_param['L23']['sst']['tau_m'], neural_param['L23']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['exc_sst'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.244}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 56.2, neural_param['L23']['sst']['tau_m'], neural_param['L23']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['sst_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.465}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L23']['sst']['tau_m'], neural_param['L23']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['sst_sst'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.244, 17.9, neural_param['L23']['vip']['tau_m'], neural_param['L23']['vip']['R']) # L23_exc to L23_vip, avermann 2011
conn_param['L23_L23']['exc_vip'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.244}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 56.2, neural_param['L23']['vip']['tau_m'], neural_param['L23']['vip']['R']) # L23_exc to L23_vip, avermann 2011
conn_param['L23_L23']['vip_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.465}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L23']['vip']['tau_m'], neural_param['L23']['vip']['R']) # L23_exc to L23_vip, avermann 2011
conn_param['L23_L23']['vip_vip'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L23']['sst']['tau_m'], neural_param['L23']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst
conn_param['L23_L23']['vip_sst'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 33.3, neural_param['L23']['pv']['tau_m'], neural_param['L23']['pv']['R']) # avermann 2011, wrong half-width
conn_param['L23_L23']['vip_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.379}

#### inside L4
conn_param['L4_L4'] = {}


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.95, 50.0, neural_param['L4']['exc']['tau_m'], neural_param['L4']['exc']['R']) # L4_exc to L4_exc, lefort 2009
conn_param['L4_L4']['exc_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.243} 

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.82, 13.7, neural_param['L4']['pv']['tau_m'], neural_param['L4']['pv']['R']) # L4_exc to L4_pv, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['exc_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.69}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.52, 43.1, neural_param['L4']['exc']['tau_m'], neural_param['L4']['exc']['R']) # L4_pv to L4_exc, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['pv_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.67}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.56, 15.8, neural_param['L4']['pv']['tau_m'], neural_param['L4']['pv']['R']) # L4_pv to L4_pv, all copy from L2/3
conn_param['L4_L4']['pv_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.55}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.244, 17.9, neural_param['L4']['sst']['tau_m'], neural_param['L4']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['exc_sst'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.40}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 56.2, neural_param['L4']['sst']['tau_m'], neural_param['L4']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst, copy from L2/3, except p adopted from Taro's presentation
conn_param['L4_L4']['sst_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.29}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.49, 33.3, neural_param['L4']['sst']['tau_m'], neural_param['L4']['sst']['R']) # nfs paramteres of avermann 2011 instead of sst, all copy from L2/3
conn_param['L4_L4']['sst_sst'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.381}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 20.0, neural_param['L4']['pv']['tau_m'], neural_param['L4']['pv']['R']) # L4_sst to L4_pv, actually vip to pv of L23, p adopted from Taro's presentation
conn_param['L4_L4']['sst_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.63}


#### L4 to L2/3
conn_param['L4_L23'] = {}

tmp_tau, tmp_w = preprocessing.Fit_PSP(0.6, 31.0, neural_param['L23']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L4_exc to L23_exc, lefort 2009
conn_param['L4_L23']['exc_exc'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.13} 
print('exc', 0.6, tmp_w)


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.50, 31.0, neural_param['L23']['pv']['tau_m'], neural_param['L23']['pv']['R']) # L4_exc to L23_exc, lefort 2009
conn_param['L4_L23']['exc_pv'] = {'w':tmp_w, 'tau_syn':tmp_tau, 'p':0.2} 
print('inh', 0.5, tmp_w) 


