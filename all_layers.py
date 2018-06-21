'''
    Master's thesis
    Last modified : 11 Jun 2018 
'''

from brian2 import *
import numpy
import matplotlib.pyplot as plt
import math
import params
import connectivity
import analysis
import os
import pickle
import preprocessing

############### simulation parameters

save_data = False
voltage_recording = False
dir = "all_layers_new"
run_counter = 'test'

if save_data:
    if not os.path.exists('result_' + dir):
            os.makedirs('result_' + dir)
            
    saving_dir='result_' + dir+'/'+str(run_counter)+'/'
    if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)


time = 1000 #ms
param_set_exc = params.param_set_exc
param_set_nfs  = params.param_set_nfs
param_set_fs = params.param_set_fs

## tested params, might have to be changed ##

exc_param = numpy.loadtxt('data/exc.txt', delimiter = ',')[param_set_exc]
nfs_param = numpy.loadtxt('data/nfs.txt', delimiter = ',')[param_set_nfs]
fs_param  = numpy.loadtxt('data/fs.txt' , delimiter = ',')[param_set_fs]


############### neural parameters

### 1---> exc neuron

C_1 = exc_param[0] * nF
gl_1 = exc_param[1] * uS
El_1 = exc_param[2] * mV
v_reset_1 = exc_param[3] * mV
tau_ref_1 = exc_param[4] * ms
v0_1 = exc_param[5] * mV
deltaV_1 = exc_param[6] * mV

amp_w1_1 = exc_param[7] * nA
tau_w1_1 = exc_param[8] * ms
amp_w2_1 = exc_param[9] * nA
tau_w2_1 = exc_param[10] * ms


amp_vt1_1 = exc_param[11] * mV
tau_vt1_1 = exc_param[12] * ms
amp_vt2_1 = exc_param[13] * mV
tau_vt2_1 = exc_param[14] * ms

rate0_1 = defaultclock.dt/ms * Hz * 100

tau_exc_1 = params.conn_param['L23_L23']['exc_exc']['tau_syn'] * ms
tau_inh_1 = params.conn_param['L23_L23']['pv_exc']['tau_syn'] * ms


eqs_1 = '''
dv/dt = (-gl_1*(v-El_1)-w1-w2+I)/C_1 : volt
dw1/dt = -w1/tau_w1_1 : amp
dw2/dt = -w2/tau_w2_1 : amp
dvt1/dt = -vt1/tau_vt1_1 : volt
dvt2/dt = -vt2/tau_vt2_1 : volt
vt = v0_1 + vt1 + vt2 : volt
rate = rate0_1*exp((v-vt)/deltaV_1): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
dI_noise/dt = -I_noise/tau_exc_1 : amp
dI_exc/dt = -I_exc/tau_exc_1 : amp
dI_inh/dt = -I_inh/tau_inh_1 : amp
'''
#w1/2 for spike triggered 
#vt1/2 for dynamic threshold doesnt change with voltage

reset_1='''
v=v_reset_1
w1+=amp_w1_1
w2+=amp_w2_1
vt1+=amp_vt1_1
vt2+=amp_vt2_1
'''

### 2 ---> pv (fs) neuron

C_2 = fs_param[0] * nF
gl_2 = fs_param[1] * uS
El_2 = fs_param[2] * mV
v_reset_2 = fs_param[3] * mV
tau_ref_2 = fs_param[4] * ms
v0_2 = fs_param[5] * mV
deltaV_2 = fs_param[6] * mV

amp_w1_2 = fs_param[7] * nA
tau_w1_2 = fs_param[8] * ms
amp_w2_2 = fs_param[9] * nA
tau_w2_2 = fs_param[10] * ms


amp_vt1_2 = fs_param[11] * mV
tau_vt1_2 = fs_param[12] * ms
amp_vt2_2 = fs_param[13] * mV
tau_vt2_2 = fs_param[14] * ms

rate0_2 = defaultclock.dt/ms * Hz * 1000

tau_exc_2 = params.conn_param['L23_L23']['exc_pv']['tau_syn'] * ms
tau_inh_2 = params.conn_param['L23_L23']['pv_pv']['tau_syn'] * ms


eqs_2 = '''
dv/dt = (-gl_2*(v-El_2)-w1-w2+I)/C_2 : volt
dw1/dt = -w1/tau_w1_2 : amp
dw2/dt = -w2/tau_w2_2 : amp
dvt1/dt = -vt1/tau_vt1_2 : volt
dvt2/dt = -vt2/tau_vt2_2 : volt
vt = v0_2 + vt1 + vt2 : volt
rate = rate0_1*exp((v-vt)/deltaV_2): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
dI_noise/dt = -I_noise/tau_exc_2 : amp
dI_exc/dt = -I_exc/tau_exc_2 : amp
dI_inh/dt = -I_inh/tau_inh_2 : amp
'''

reset_2='''
v=v_reset_2
w1+=amp_w1_2
w2+=amp_w2_2
vt1+=amp_vt1_2
vt2+=amp_vt2_2
'''

### 3 ---> sst neuron, but we use nfs neuron parameters for that

C_3 = nfs_param[0] * nF
gl_3 = nfs_param[1] * uS
El_3 = nfs_param[2] * mV
v_reset_3 = nfs_param[3] * mV
tau_ref_3 = nfs_param[4] * ms
v0_3 = nfs_param[5] * mV
deltaV_3 = nfs_param[6] * mV

amp_w1_3 = nfs_param[7] * nA
tau_w1_3 = nfs_param[8] * ms
amp_w2_3 = nfs_param[9] * nA
tau_w2_3 = nfs_param[10] * ms


amp_vt1_3 = nfs_param[11] * mV
tau_vt1_3 = nfs_param[12] * ms
amp_vt2_3 = nfs_param[13] * mV
tau_vt2_3 = nfs_param[14] * ms

rate0_3 = defaultclock.dt/ms * Hz * 1000

tau_exc_3 = params.conn_param['L23_L23']['exc_sst']['tau_syn'] * ms     # nfs instead of sst
tau_inh_3 = params.conn_param['L23_L23']['sst_sst']['tau_syn'] * ms     # nfs instead of sst


eqs_3 = '''
dv/dt = (-gl_3*(v-El_3)-w1-w2+I)/C_3 : volt
dw1/dt = -w1/tau_w1_3 : amp
dw2/dt = -w2/tau_w2_3 : amp
dvt1/dt = -vt1/tau_vt1_3 : volt
dvt2/dt = -vt2/tau_vt2_3 : volt
vt = v0_3 + vt1 + vt2 : volt
rate = rate0_1*exp((v-vt)/deltaV_3): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
dI_noise/dt = -I_noise/tau_exc_3 : amp
dI_exc/dt = -I_exc/tau_exc_3 : amp
dI_inh/dt = -I_inh/tau_inh_3 : amp
'''

reset_3='''
v=v_reset_3
w1+=amp_w1_3
w2+=amp_w2_3
vt1+=amp_vt1_3
vt2+=amp_vt2_3
'''

### 4 ---> vip neuron, but do we also use nfs params for that

C_4 = nfs_param[0] * nF
gl_4 = nfs_param[1] * uS
El_4 = nfs_param[2] * mV
v_reset_4 = nfs_param[3] * mV
tau_ref_4 = nfs_param[4] * ms
v0_4 = nfs_param[5] * mV
deltaV_4 = nfs_param[6] * mV

amp_w1_4 = nfs_param[7] * nA
tau_w1_4 = nfs_param[8] * ms
amp_w2_4 = nfs_param[9] * nA
tau_w2_4 = nfs_param[10] * ms


amp_vt1_4 = nfs_param[11] * mV
tau_vt1_4 = nfs_param[12] * ms
amp_vt2_4 = nfs_param[13] * mV
tau_vt2_4 = nfs_param[14] * ms

rate0_4 = defaultclock.dt/ms * Hz * 1000

tau_exc_4 = params.conn_param['L23_L23']['exc_vip']['tau_syn'] * ms
tau_inh_4 = params.conn_param['L23_L23']['vip_vip']['tau_syn'] * ms

# w = eta
# vt= gamma

eqs_4 = '''
dv/dt = (-gl_4*(v-El_4)-w1-w2+I)/C_4 : volt
dw1/dt = -w1/tau_w1_4 : amp
dw2/dt = -w2/tau_w2_4 : amp
dvt1/dt = -vt1/tau_vt1_4 : volt
dvt2/dt = -vt2/tau_vt2_4 : volt
vt = v0_4 + vt1 + vt2 : volt
rate = rate0_1*exp((v-vt)/deltaV_4): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
dI_noise/dt = -I_noise/tau_exc_4 : amp
dI_exc/dt = -I_exc/tau_exc_4 : amp
dI_inh/dt = -I_inh/tau_inh_4 : amp
'''

reset_4='''
v=v_reset_4
w1+=amp_w1_4
w2+=amp_w2_4
vt1+=amp_vt1_4
vt2+=amp_vt2_4
'''


############## forming populations



