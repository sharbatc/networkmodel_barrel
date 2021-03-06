'''
    Master's project
    Containing parameters from papers, Carl, Hesam
    Sharbat, last updated : April 22, 2018
'''
import preprocessing
import numpy


param_set_exc = 3 # 2
param_set_nfs = 2 
param_set_fs  = 1 # 0
param_set_sst  = 1


exc_param = numpy.loadtxt('data/exc.txt', delimiter = ',')[param_set_exc]
nfs_param = numpy.loadtxt('data/nfs.txt', delimiter = ',')[param_set_nfs]
fs_param  = numpy.loadtxt('data/fs.txt' , delimiter = ',')[param_set_fs]
sst_param  = numpy.loadtxt('data/nfs.txt' , delimiter = ',')[param_set_sst] # nfs instead of sst


###### size of the populations
# important: we don't implement non-vip neuron 

# these are old numbers
#L23_size = {'exc':1691, 'sst':46, 'vip':46, 'pv': 69}  # by lefort, exc: 1691, inh: 230, 30% for pv, 20% for sst, 50% for nfs (40% for vip, 60% for non-vip)
#L4_size  = {'exc':1656, 'sst':84, 'vip':0, 'pv': 56}

# New numbers, exc from Lefort, Other numbers from Taro experiment (real and experimentally extracted numbers)
L23_size = {'exc':1691, 'sst':67, 'vip':95, 'pv': 101}
L4_size  = {'exc':1656, 'sst':21, 'vip':7, 'pv': 87} # we do not implement vip neurons for L4


size = {'L23':L23_size, 'L4':L4_size}


###### neural parameters

#L23_exc= {'tau_m':28.4, 'R':160, 'tau_ref':4, 'v_thresh':-37.9, 'v_reset':-68.1, 'v_rest':-68.1} # avermann 2011
#L23_vip= {'tau_m':21.2, 'R':208, 'tau_ref':4, 'v_thresh':-36.3, 'v_reset':-62.6, 'v_rest':-62.6} # avermann 2011, NFS
#L23_pv = {'tau_m':9.3 , 'R':99 , 'tau_ref':4, 'v_thresh':-37.4, 'v_reset':-67.5, 'v_rest':-67.5} # avermann 2011, FS
#L23_sst= {'tau_m':15, 'R':150, 'tau_ref':4, 'v_thresh':-37, 'v_reset':-61, 'v_rest':-61} # lee 2013 nature, only contains R and v_rest

L23_exc= {}
L23_vip= {}
L23_pv = {}
L23_sst= {}


L23_exc['R'] = 1.0/exc_param[1]
L23_pv['R'] = 1.0/fs_param[1]
L23_sst['R'] = 1.0/nfs_param[1]
L23_vip['R'] = 1.0/nfs_param[1]

L23_exc['tau_m'] = exc_param[0]/exc_param[1]
L23_vip['tau_m'] = nfs_param[0]/nfs_param[1]
L23_sst['tau_m'] = sst_param[0]/sst_param[1]
L23_pv['tau_m'] = fs_param[0]/fs_param[1]



#L4_exc = {'tau_m':34.8, 'R':302, 'tau_ref':4, 'v_thresh':-39.7, 'v_reset':-66, 'v_rest':-66} # lefort 2009
#L4_vip= {'tau_m':21.2, 'R':208, 'tau_ref':4, 'v_thresh':-36.3, 'v_reset':-62.6, 'v_rest':-62.6} # copied from L23
#L4_pv = {'tau_m':9.3 , 'R':99 , 'tau_ref':4, 'v_thresh':-37.4, 'v_reset':-67.5, 'v_rest':-67.5} # copied from L23
#L4_sst= {'tau_m':15, 'R':150, 'tau_ref':4, 'v_thresh':-37, 'v_reset':-61, 'v_rest':-61} # copied from L23

L4_exc = {}
L4_pv = {}
L4_sst= {}


L4_exc['R'] = 1.0/exc_param[1]
L4_pv['R'] = 1.0/fs_param[1]
L4_sst['R'] = 1.0/nfs_param[1]

L4_exc['tau_m'] = exc_param[0]/exc_param[1]
L4_sst['tau_m'] = sst_param[0]/sst_param[1]
L4_pv['tau_m'] = fs_param[0]/fs_param[1]


L23_neural = {'exc':L23_exc, 'sst':L23_sst, 'vip':L23_vip, 'pv':L23_pv}
L4_neural  = {'exc':L4_exc, 'sst':L4_sst, 'pv':L4_pv}

neural_param = {'L23':L23_neural, 'L4':L4_neural}

###################### connection parameters

conn_param = {'L23_L23':None, 'L4_L4':None, 'L4_L23':None}

#### inside L2/3
conn_param['L23_L23'] = {}


tmp_tau, tmp_w = preprocessing.Fit_PSP(0.37, 26.2, neural_param['L23']['exc']['tau_m'], neural_param['L23']['exc']['R']) # L23_exc to L23_exc, avermann 2011
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