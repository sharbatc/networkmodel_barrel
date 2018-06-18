'''
    Project Disinhibition
    NEW VERSION (new pop. sizes and new network regulation)
    Implementation of L4 and L2/3 using GIF
    L4 contains exc, pv(fs) and stt, but not vip neurons
    L2/3 contains exc, pv(fs) and stt and vip neurons
    Started : 4 July, 2017 by Hesam SETAREH
    Last modified : 11 Jun, 2017 by Sharbat 
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


def step(x):
    if x < 0:
        return 0
    else:
        return 1
    
def multistep(x):
    return int(x)%2


############### simulation parameters

save_data = False
voltage_recording = False
dir = "two_layers_new"
run_counter = 'test'

if save_data:
    if not os.path.exists('result_' + dir):
            os.makedirs('result_' + dir)
            
    saving_dir='result_' + dir+'/'+str(run_counter)+'/'
    if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)


time = 3000 #ms
param_set_exc = params.param_set_exc
param_set_nfs  = params.param_set_nfs
param_set_fs = params.param_set_fs

## tested params, might have to be changed ##

exc_param = numpy.loadtxt('data/exc.txt', delimiter = ',')[param_set_exc]
nfs_param = numpy.loadtxt('data/nfs.txt', delimiter = ',')[param_set_nfs]
fs_param  = numpy.loadtxt('data/fs.txt' , delimiter = ',')[param_set_fs]


############### neural parameters

# reversal potential for COBA synapses
E_exc = 0 * mV
E_inh = -80 * mV

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

lambda0_1 = defaultclock.dt/ms * Hz * 100

tau_exc_1 = params.conn_param['L23_L23']['exc_exc']['tau_syn'] * ms
tau_inh_1 = params.conn_param['L23_L23']['pv_exc']['tau_syn'] * ms


# w = eta
# vt= gamma

eqs_1 = '''
dv/dt = (-gl_1*(v-El_1)-w1-w2+I)/C_1 : volt
dw1/dt = -w1/tau_w1_1 : amp
dw2/dt = -w2/tau_w2_1 : amp
dvt1/dt = -vt1/tau_vt1_1 : volt
dvt2/dt = -vt2/tau_vt2_1 : volt
vt = v0_1 + vt1 + vt2 : volt
lambda = lambda0_1*exp((v-vt)/deltaV_1): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
I_noise = -g_noise*(v-E_exc) : amp
I_exc = -g_exc*(v-E_exc) : amp
I_inh = -g_inh*(v-E_inh) : amp
dg_noise/dt = -g_noise/tau_exc_1 : siemens
dg_exc/dt = -g_exc/tau_exc_1 : siemens
dg_inh/dt = -g_inh/tau_inh_1 : siemens
'''

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

lambda0_2 = defaultclock.dt/ms * Hz * 1000

tau_exc_2 = params.conn_param['L23_L23']['exc_pv']['tau_syn'] * ms
tau_inh_2 = params.conn_param['L23_L23']['pv_pv']['tau_syn'] * ms

# w = eta
# vt= gamma

eqs_2 = '''
dv/dt = (-gl_2*(v-El_2)-w1-w2+I)/C_2 : volt
dw1/dt = -w1/tau_w1_2 : amp
dw2/dt = -w2/tau_w2_2 : amp
dvt1/dt = -vt1/tau_vt1_2 : volt
dvt2/dt = -vt2/tau_vt2_2 : volt
vt = v0_2 + vt1 + vt2 : volt
lambda = lambda0_1*exp((v-vt)/deltaV_2): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
I_noise = -g_noise*(v-E_exc) : amp
I_exc = -g_exc*(v-E_exc) : amp
I_inh = -g_inh*(v-E_inh) : amp
dg_noise/dt = -g_noise/tau_exc_2 : siemens
dg_exc/dt = -g_exc/tau_exc_2 : siemens
dg_inh/dt = -g_inh/tau_inh_2 : siemens
'''

reset_2='''
v=v_reset_2
w1+=amp_w1_2
w2+=amp_w2_2
vt1+=amp_vt1_2
vt2+=amp_vt2_2
'''



### 3 ---> sst neuron, but we use nfs (vip) neuron parameters for that

C_3 = sst_param[0] * nF
gl_3 = sst_param[1] * uS
El_3 = sst_param[2] * mV
v_reset_3 = sst_param[3] * mV
tau_ref_3 = sst_param[4] * ms
v0_3 = sst_param[5] * mV
deltaV_3 = sst_param[6] * mV

amp_w1_3 = sst_param[7] * nA
tau_w1_3 = sst_param[8] * ms
amp_w2_3 = sst_param[9] * nA
tau_w2_3 = sst_param[10] * ms


amp_vt1_3 = sst_param[11] * mV
tau_vt1_3 = sst_param[12] * ms
amp_vt2_3 = sst_param[13] * mV
tau_vt2_3 = sst_param[14] * ms

lambda0_3 = defaultclock.dt/ms * Hz * 1000

tau_exc_3 = params.conn_param['L23_L23']['exc_sst']['tau_syn'] * ms     # nfs instead of sst
tau_inh_3 = params.conn_param['L23_L23']['sst_sst']['tau_syn'] * ms     # nfs instead of sst

# w = eta
# vt= gamma

eqs_3 = '''
dv/dt = (-gl_3*(v-El_3)-w1-w2+I)/C_3 : volt
dw1/dt = -w1/tau_w1_3 : amp
dw2/dt = -w2/tau_w2_3 : amp
dvt1/dt = -vt1/tau_vt1_3 : volt
dvt2/dt = -vt2/tau_vt2_3 : volt
vt = v0_3 + vt1 + vt2 : volt
lambda = lambda0_1*exp((v-vt)/deltaV_3): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
I_noise = -g_noise*(v-E_exc) : amp
I_exc = -g_exc*(v-E_exc) : amp
I_inh = -g_inh*(v-E_inh) : amp
dg_noise/dt = -g_noise/tau_exc_3 : siemens
dg_exc/dt = -g_exc/tau_exc_3 : siemens
dg_inh/dt = -g_inh/tau_inh_3 : siemens
'''

reset_3='''
v=v_reset_3
w1+=amp_w1_3
w2+=amp_w2_3
vt1+=amp_vt1_3
vt2+=amp_vt2_3
'''


### 4 ---> vip neuron

C_4 = vip_param[0] * nF
gl_4 = vip_param[1] * uS
El_4 = vip_param[2] * mV
v_reset_4 = vip_param[3] * mV
tau_ref_4 = vip_param[4] * ms
v0_4 = vip_param[5] * mV
deltaV_4 = vip_param[6] * mV

amp_w1_4 = vip_param[7] * nA
tau_w1_4 = vip_param[8] * ms
amp_w2_4 = vip_param[9] * nA
tau_w2_4 = vip_param[10] * ms


amp_vt1_4 = vip_param[11] * mV
tau_vt1_4 = vip_param[12] * ms
amp_vt2_4 = vip_param[13] * mV
tau_vt2_4 = vip_param[14] * ms

lambda0_4 = defaultclock.dt/ms * Hz * 1000

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
lambda = lambda0_1*exp((v-vt)/deltaV_4): Hz
I = I_syn + I_ext : amp
I_ext: amp
I_syn = I_exc + I_inh + I_noise : amp
I_noise = -g_noise*(v-E_exc) : amp
I_exc = -g_exc*(v-E_exc) : amp
I_inh = -g_inh*(v-E_inh) : amp
dg_noise/dt = -g_noise/tau_exc_4 : siemens
dg_exc/dt = -g_exc/tau_exc_4 : siemens
dg_inh/dt = -g_inh/tau_inh_4 : siemens
'''

reset_4='''
v=v_reset_4
w1+=amp_w1_4
w2+=amp_w2_4
vt1+=amp_vt1_4
vt2+=amp_vt2_4
'''



############## forming populations
pops = {}

pops['L4exc'] = NeuronGroup(params.size['L4']['exc'], model=eqs_1, reset = reset_1, threshold ='rand()<rate*dt', refractory=tau_ref_1)
pops['L4pv'] = NeuronGroup(params.size['L4']['pv'], model=eqs_2, reset = reset_2, threshold ='rand()<rate*dt', refractory=tau_ref_2)
pops['L4sst'] = NeuronGroup(params.size['L4']['sst'], model=eqs_3, reset = reset_3, threshold ='rand()<rate*dt', refractory=tau_ref_3)

pops['L23exc'] = NeuronGroup(params.size['L23']['exc'], model=eqs_1, reset = reset_1, threshold ='rand()<rate*dt', refractory=tau_ref_1)
pops['L23pv'] = NeuronGroup(params.size['L23']['pv'], model=eqs_2, reset = reset_2, threshold ='rand()<rate*dt', refractory=tau_ref_2)
pops['L23sst'] = NeuronGroup(params.size['L23']['sst'], model=eqs_3, reset = reset_3, threshold ='rand()<rate*dt', refractory=tau_ref_3)
pops['L23vip'] = NeuronGroup(params.size['L23']['vip'], model=eqs_4, reset = reset_4, threshold ='rand()<rate*dt', refractory=tau_ref_4)


#thalamus = PoissonGroup(600, rates = lambda t:(1+math.sin(t*5))*7*Hz)

#thalamus = PoissonGroup(600, rates = lambda t:heavyside(math.sin(t*5))*9*Hz)

#thalamus = PoissonGroup(600, rates = lambda t:(step(t-1000*ms)-step(t-2000*ms))*15*Hz)
# rates_poisson = '((t/second)%2)*15*Hz)' 
# thalamus = PoissonGroup(600, rates = rates_poisson) 


########################## initial values
pops['L4exc'].v = exc_param[2] * mV
pops['L4pv'].v =  fs_param[2] * mV
pops['L4sst'].v =  sst_param[2] * mV

pops['L23exc'].v = exc_param[2] * mV
pops['L23pv'].v =  fs_param[2] * mV
pops['L23sst'].v =  sst_param[2] * mV
pops['L23vip'].v =  vip_param[2] * mV

########################## Synapsess
####### inside L4
conn_L4exc_L4exc = Synapses(pops['L4exc'], pops['L4exc'], 'w : amp', on_pre = 'I_exc += w')
conn_L4exc_L4exc.connect(p=params.conn_param['L4_L4']['pv_pv']['p'])
conn_L4exc_L4exc.w = params.conn_param['L4_L4']['exc_exc']['w']*nA

conn_L4pv_L4pv = Synapses(pops['L4pv'], pops['L4pv'], 'w : amp', on_pre = 'I_inh += w') 
conn_L4pv_L4pv.connect(p=params.conn_param['L4_L4']['pv_pv']['p']) 
conn_L4pv_L4pv.w =1.1*-params.conn_param['L4_L4']['pv_pv']['w']*nA ##### w = 1.3

conn_L4sst_L4sst = Synapses(pops['L4sst'], pops['L4sst'], 'w : amp', on_pre = 'I_inh += w')
conn_L4sst_L4sst.connect(p=params.conn_param['L4_L4']['sst_sst']['p'])
conn_L4sst_L4sst.w = 0.8*-params.conn_param['L4_L4']['sst_sst']['w']*nA

conn_L4exc_L4pv = Synapses(pops['L4exc'], pops['L4pv'], 'w : amp', on_pre = 'I_exc += w')
conn_L4exc_L4pv.connect(p=params.conn_param['L4_L4']['exc_pv']['p'])
conn_L4exc_L4pv = 0.6*params.conn_param['L4_L4']['exc_pv']['w']*nA

conn_L4pv_L4exc = Synapses(pops['L4pv'], pops['L4exc'], 'w : amp', on_pre = 'I_inh += w')
conn_L4pv_L4exc.connect(p = params.conn_param['L4_L4']['pv_exc']['p'])
conn_L4pv_L4exc.w = 0.7*-params.conn_param['L4_L4']['pv_exc']['w']*nA

conn_L4exc_L4sst = Synapses(pops['L4exc'], pops['L4sst'], 'w : amp', on_pre = 'I_exc += w') 
conn_L4exc_L4sst.connect(p = params.conn_param['L4_L4']['exc_sst']['p'])
conn_L4exc_L4sst.w = 1.15*params.conn_param['L4_L4']['exc_sst']['w']*nA # weight = *1.1

conn_L4sst_L4exc = Synapses(pops['L4sst'], pops['L4exc'], 'w : amp', on_pre = 'I_inh += w')
conn_L4sst_L4exc.connect(p = params.conn_param['L4_L4']['sst_exc']['p']) 
conn_L4sst_L4exc.w =-params.conn_param['L4_L4']['sst_exc']['w']*nA

conn_L4sst_L4pv = Synapses(pops['L4sst'], pops['L4pv'], 'w : amp', on_pre = 'I_inh += w')
conn_L4sst_L4pv.connect(p = params.conn_param['L4_L4']['sst_pv']['p'])
conn_L4sst_L4pv.w =-params.conn_param['L4_L4']['sst_pv']['w']*nA


####### inside L2/3
conn_L23exc_L23exc = Synapses(pops['L23exc'], pops['L23exc'], 'w : amp', on_pre = 'I_exc += w')
conn_L23exc_L23exc.connect(p = params.conn_param['L23_L23']['exc_exc']['p'])
conn_L23exc_L23exc.w = params.conn_param['L23_L23']['exc_exc']['w']*nA

conn_L23pv_L23pv = Synapses(pops['L23pv'], pops['L23pv'], 'w : amp', on_pre='I_inh += w')
conn_L23pv_L23pv.connect( p = params.conn_param['L23_L23']['pv_pv']['p'])
conn_L23pv_L23pv.w =-params.conn_param['L23_L23']['pv_pv']['w']*nA

conn_L23sst_L23sst = Synapses(pops['L23sst'], pops['L23sst'], 'w : amp', on_pre = 'I_inh += w')
conn_L23sst_L23sst.connect(p=params.conn_param['L23_L23']['sst_sst']['p']) 
conn_L23sst_L23sst.w = -params.conn_param['L23_L23']['sst_sst']['w']*nA

conn_L23vip_L23vip = Synapses(pops['L23vip'], pops['L23vip'], 'w : amp', on_pre='I_inh += w')
conn_L23vip_L23vip.connect(p=params.conn_param['L23_L23']['vip_vip']['p'])
conn_L23vip_L23vip.w = 0.7*-params.conn_param['L23_L23']['vip_vip']['w']*nA

conn_L23exc_L23pv = Synapses(pops['L23exc'], pops['L23pv'], 'w : amp', on_pre='I_exc += w')
conn_L23exc_L23pv.connect( p = params.conn_param['L23_L23']['exc_pv']['p'])
conn_L23exc_L23pv.w = 0.6*params.conn_param['L23_L23']['exc_pv']['w']*nA

conn_L23pv_L23exc = Synapses(pops['L23pv'], pops['L23exc'], 'w : amp', on_pre = 'I_inh += w')
conn_L23pv_L23exc.connect(p=params.conn_param['L23_L23']['pv_exc']['p'])
conn_L23pv_L23exc.w = 0.7*-params.conn_param['L23_L23']['pv_exc']['w']*nA

conn_L23exc_L23sst = Synapses(pops['L23exc'], pops['L23sst'], 'w : amp', on_pre='I_exc += w')
conn_L23exc_L23sst.connect(p=params.conn_param['L23_L23']['exc_sst']['p'])
conn_L23exc_L23sst.w = params.conn_param['L23_L23']['exc_sst']['w']*nA

# conn_L23exc_L23sst_stp = STP(conn_L23exc_L23sst, taud=20*ms,tauf=1000*ms, U=0.05) # for parameters look at  krishnamurthy 2012 plos one
## to be checked

conn_L23sst_L23exc = Synapses(pops['L23sst'], pops['L23exc'], 'w : amp', on_pre='I_inh += w')
conn_L23sst_L23exc.connect(p=params.conn_param['L23_L23']['sst_exc']['p'])
conn_L23sst_L23exc.w =-params.conn_param['L23_L23']['sst_exc']['w']*nA


conn_L23exc_L23vip = Synapses(pops['L23exc'], pops['L23vip'], 'w : amp', on_pre='I_exc += w')
conn_L23exc_L23vip.connect(p = params.conn_param['L23_L23']['exc_vip']['p'])
conn_L23exc_L23vip.w = 1.3*params.conn_param['L23_L23']['exc_vip']['w']*nA

conn_L23vip_L23exc = Synapses(pops['L23vip'], pops['L23exc'], 'w : amp', on_pre='I_inh += w')
conn_L23vip_L23exc.connect(p =  params.conn_param['L23_L23']['vip_exc']['p'])
conn_L23vip_L23exc.w =-1.2*params.conn_param['L23_L23']['vip_exc']['w']*nA

conn_L23vip_L23sst = Synapses(pops['L23vip'], pops['L23sst'], 'w : amp', on_pre='I_inh += w')
conn_L23vip_L23sst.connect(p= 1.4*params.conn_param['L23_L23']['vip_sst']['p'])
conn_L23vip_L23sst.w =-1.5*params.conn_param['L23_L23']['vip_sst']['w']*nA

conn_L23vip_L23pv = Synapses(pops['L23vip'], pops['L23pv'], 'w : amp', on_pre = 'I_inh += w')
conn_L23vip_L23pv.connect(p = params.conn_param['L23_L23']['vip_pv']['p'])
conn_L23vip_L23pv.w =-params.conn_param['L23_L23']['vip_pv']['w']*nA


####### L4 to L2/3
conn_L4exc_L23exc = Synapses(pops['L4exc'], pops['L23exc'], 'w : amp', on_pre='I_exc += w')
conn_L4exc_L23exc.connect(p=params.conn_param['L4_L23']['exc_exc']['p'])
conn_L4exc_L23exc.w = 0.6*params.conn_param['L4_L23']['exc_exc']['w']*nA

conn_L4exc_L23pv = Synapses(pops['L4exc'], pops['L23pv'], 'w : amp', on_pre='I_exc += w')
conn_L4exc_L23pv.connect(p=params.conn_param['L4_L23']['exc_pv']['p'])
conn_L4exc_L23pv.w = 0.8*params.conn_param['L4_L23']['exc_pv']['w']*nA

conn_L4exc_L23vip = Synapses(pops['L4exc'], pops['L23vip'], 'w : amp', on_pre='I_exc += w')
conn_L4exc_L23vip.connect(p=params.conn_param['L4_L23']['exc_pv']['p'])
conn_L4exc_L23vip.w = params.conn_param['L4_L23']['exc_pv']['w']*nA


# L2 - L3
# L5a, 5b, 6
# excitatory purely

####### from Thalamus - not sure where they come from, ask Semih
# conn_th_L4exc = Synapses(thalamus, pops['L4exc'], sparseness=0.2, weight = 0.02*nA, state='I_exc')  # w should be fixed
# conn_th_L4pv = Synapses(thalamus, pops['L4pv'], sparseness=0.10, weight = 0.01*nA, state='I_exc')  # w should be fixed

# conn_th_L23exc = Synapses(thalamus, pops['L23exc'], sparseness=0.05, weight = 0.02*nA, state='I_exc')  # w should be fixed
# conn_th_L23pv = Synapses(thalamus, pops['L23pv'], sparseness=0.05, weight = 0.01*nA, state='I_exc')  # w should be fixed


###### noise

# noise_L4exc  = PoissonInput(pops['L4exc'], 600, rate=10*Hz, weight=0.015*nA, state='I_noise')  # poisson input to L4_exc # 600, 10
# noise_L4pv = PoissonInput(pops['L4pv'], 500, rate=5*Hz, weight=0.015*nA, state='I_noise')    # poisson input to L4_pv # 400, 5
# noise_L4sst = PoissonInput(pops['L4sst'], 250, rate=5*Hz, weight=0.015*nA, state='I_noise')   # poisson input to L4_sst


# noise_L23exc  = PoissonInput(pops['L23exc'], 600, rate=10*Hz, weight=0.01*nA, state='I_noise')  # poisson input to L4_exc # 600, 10
# noise_L23pv = PoissonInput(pops['L23pv'], 500, rate=5*Hz, weight=0.015*nA, state='I_noise')    # poisson input to L4_pv # 400, 5
# noise_L23sst = PoissonInput(pops['L23sst'], 250, rate=5*Hz, weight=0.015*nA, state='I_noise')   # poisson input to L4_sst
# noise_L23vip = PoissonInput(pops['L23vip'], 100, rate=5*Hz, weight=0.015*nA, state='I_noise')    # poisson input to L4_pv # 400, 5


############## recorders
# rec_I_noise = {}
rec_I_exc = {}
rec_I_inh = {}


'''
rec_I_noise['sst'] = StateMonitor(L4_sst, 'I_noise', record = True)
rec_I_exc['sst'] = StateMonitor(L4_sst, 'I_exc', record = True)
rec_I_inh['sst'] = StateMonitor(L4_sst, 'I_inh', record = True)

rec_I_noise['pv'] = StateMonitor(L4_pv, 'I_noise', record = True)
rec_I_exc['pv'] = StateMonitor(L4_pv, 'I_exc', record = True)
rec_I_inh['pv'] = StateMonitor(L4_pv, 'I_inh', record = True)

rec_I_noise['exc'] = StateMonitor(L4_exc, 'I_noise', record = True)
rec_I_exc['exc'] = StateMonitor(L4_exc, 'I_exc', record = True)
rec_I_inh['exc'] = StateMonitor(L4_exc, 'I_inh', record = True)
'''

# rec_I_noise['L23vip'] = StateMonitor(pops['L23vip'], 'I_noise', record = True)
rec_I_exc['L23vip'] = StateMonitor(pops['L23vip'], 'I_exc', record = True)
rec_I_inh['L23vip'] = StateMonitor(pops['L23vip'], 'I_inh', record = True)


# rec_I_noise['L23sst'] = StateMonitor(pops['L23sst'], 'I_noise', record = True)
rec_I_exc['L23sst'] = StateMonitor(pops['L23sst'], 'I_exc', record = True)
rec_I_inh['L23sst'] = StateMonitor(pops['L23sst'], 'I_inh', record = True)

# rec_I_noise['L4sst'] = StateMonitor(pops['L4sst'], 'I_noise', record = True)
rec_I_exc['L4sst'] = StateMonitor(pops['L4sst'], 'I_exc', record = True)
rec_I_inh['L4sst'] = StateMonitor(pops['L4sst'], 'I_inh', record = True)


if voltage_recording:
    rec_v = {}
    
    rec_v['L4exc'] = StateMonitor(pops['L4exc'], 'v', record = True)
    rec_v['L4pv'] = StateMonitor(pops['L4pv'], 'v', record = True)
    rec_v['L4sst'] = StateMonitor(pops['L4sst'], 'v', record = True)
    
    rec_v['L23exc'] = StateMonitor(pops['L23exc'], 'v', record = True)
    rec_v['L23pv'] = StateMonitor(pops['L23pv'], 'v', record = True)
    rec_v['L23sst'] = StateMonitor(pops['L23sst'], 'v', record = True)
    rec_v['L23vip'] = StateMonitor(pops['L23vip'], 'v', record = True)


spikes = {}
poprate = {} # population rate monitor

spikecount = {} # number of spike for each neuron

spikecount['L4exc'] = SpikeMonitor(pops['L4exc'], record = False)
spikecount['L4sst'] = SpikeMonitor(pops['L4sst'], record = False)
spikecount['L4pv'] = SpikeMonitor(pops['L4pv'], record = False)

spikecount['L23exc'] = SpikeMonitor(pops['L23exc'], record = False)
spikecount['L23sst'] = SpikeMonitor(pops['L23sst'], record = False)
spikecount['L23pv'] = SpikeMonitor(pops['L23pv'], record = False)
spikecount['L23vip'] = SpikeMonitor(pops['L23vip'], record = False)


# spikes['thalamus'] = SpikeMonitor(thalamus)
spikes['L4exc'] = SpikeMonitor(pops['L4exc'], record = False)
spikes['L4pv'] = SpikeMonitor(pops['L4pv'], record = False)
spikes['L4sst'] = SpikeMonitor(pops['L4sst'], record = False)

spikes['L23exc'] = SpikeMonitor(pops['L23exc'], record = False)
spikes['L23pv'] = SpikeMonitor(pops['L23pv'], record = False)
spikes['L23sst'] = SpikeMonitor(pops['L23sst'], record = False)
spikes['L23vip'] = SpikeMonitor(pops['L23vip'], record = False)


# poprate['thalamus'] = SpikeMonitor(thalamus, record = False)
poprate['L4exc'] = SpikeMonitor(pops['L4exc'], record = False)
poprate['L4pv'] = SpikeMonitor(pops['L4pv'], record = False)
poprate['L4sst'] = SpikeMonitor(pops['L4sst'], record = False)

poprate['L23exc'] = SpikeMonitor(pops['L23exc'], record = False)
poprate['L23pv'] = SpikeMonitor(pops['L23pv'], record = False)
poprate['L23sst'] = SpikeMonitor(pops['L23sst'], record = False)
poprate['L23vip'] = SpikeMonitor(pops['L23vip'], record = False)

############## simulation
run(time *ms, report='text')



############# data analysis


############# plot
'''
plt.figure()
raster_plot(spikes['thalamus'])
plt.title('thalamus')

plt.figure()
plt.title('thalamus rate')
plt.plot(poprate['thalamus'].times/ms, poprate['thalamus'].rate, lw=2)
plt.grid()
'''


for label in ['L4sst', 'L23sst']:
    
    plt.figure()
    plt.title('synaptic current ' + label)
    # mean_I_noise = numpy.mean(rec_I_noise[label].values, axis=0)/nA
    mean_I_exc = numpy.mean(rec_I_exc[label].I_exc, axis=0)/nA
    mean_I_inh = numpy.mean(rec_I_inh[label].I_inh, axis=0)/nA
    mean_I_total = mean_I_exc + mean_I_inh
    # plt.plot(rec_I_exc[label].times/ms, mean_I_noise, 'green', lw=2, label='noise')
    plt.plot(rec_I_exc[label].t/ms, mean_I_exc, 'blue', lw=2, label='Exc')
    plt.plot(rec_I_exc[label].t/ms, mean_I_inh, 'red', lw=2, label='Inh')
    plt.plot(rec_I_exc[label].t/ms, mean_I_total, 'black', lw=2, label='Total')
    plt.legend()
    plt.grid()
    plt.xlabel('time (ms)')
    plt.ylabel('current (nA)')


AP_shape = numpy.zeros(1)
AP_shape[0] = 10


if voltage_recording:
    
    for label in ['L4sst', 'L4pv', 'L4exc']:
        
        plt.figure()   
        v0 = rec_v[label][0]/mV
        v1 = rec_v[label][1]/mV
        analysis.replace_AP(v0, spikes[label].spiketimes[0], AP_shape)
        analysis.replace_AP(v1, spikes[label].spiketimes[1], AP_shape)
        plt.subplot(211)
        plt.title('voltage - L4 ' + label)
        plt.plot(rec_v[label].t/ms, v0, 'blue', lw=2)
        plt.subplot(212)
        plt.plot(rec_v[label].t/ms, v1, 'red', lw=2)
        plt.grid()
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')

'''
plt.figure()
raster_plot(spikes['L4exc'])
plt.title('L4exc - raster')

plt.figure()
raster_plot(spikes['L4pv'])
plt.title('L4pv - raster')
'''
# plt.figure()
# raster_plot(spikes['L4sst'])
# plt.title('L4sst - raster')


# plt.figure()
# raster_plot(spikes['L23exc'])
# plt.title('L23exc - raster')

# plt.figure()
# raster_plot(spikes['L23pv'])
# plt.title('L23pv - raster')

# plt.figure()
# raster_plot(spikes['L23sst'])
# plt.title('L23sst - raster')

# plt.figure()
# raster_plot(spikes['L23vip'])
# plt.title('L23vip - raster')


############ not correct
#print "\nRates of L4:"
#for label in ['exc', 'pv', 'sst']:
#    print label + ' rate, before: ' + str(numpy.mean(poprate['L4'+label].rate[10:50])) + ', after: ' + str(numpy.mean(poprate['L4'+label].rate[50:])) 

#print "\nRates of L2/3:"
#for label in ['exc', 'pv', 'vip', 'sst']:
#    print label + ' rate, before: ' + str(numpy.mean(poprate['L23'+label].rate[10:50])) + ', after: ' + str(numpy.mean(poprate['L23'+label].rate[50:])) 


plt.figure()
plt.plot(poprate['L4exc'].t/ms, poprate['L4exc'].rate, label='exc', lw=2)
plt.plot(poprate['L4pv'].t/ms, poprate['L4pv'].rate, label='pv', lw=2)
plt.plot(poprate['L4sst'].t/ms, poprate['L4sst'].rate, label='sst', lw=2)
plt.legend()
plt.grid()
plt.title('L4 - firing rates')
plt.xlabel('time (ms)')
plt.ylabel('firing rate (Hz)')


plt.figure()
plt.plot(poprate['L23exc'].t/ms, poprate['L23exc'].rate, label='exc', lw=2)
plt.plot(poprate['L23pv'].t/ms, poprate['L23pv'].rate, label='pv', lw=2)
plt.plot(poprate['L23sst'].t/ms, poprate['L23sst'].rate, label='sst', lw=2)
plt.plot(poprate['L23vip'].t/ms, poprate['L23vip'].rate, label='vip', lw=2)
plt.legend()
plt.grid()
plt.title('L2/3 - firing rates')
plt.xlabel('time (ms)')
plt.ylabel('firing rate (Hz)')

############# save

if save_data:
    
    for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
        output = open(saving_dir+'/spikes_' + label  + '.txt', 'w')
        pickle.dump(spikes[label].spiketimes, output)
        output.close()

    if(voltage_recording):
        numpy.save(saving_dir+'/voltages_times.txt', rec_v['L4exc'].t/ms)
        for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
            numpy.save(saving_dir+'/voltages_' + label + '.txt', rec_v[label].v/mV)
        
    for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
        numpy.savetxt(saving_dir+'/spikecount_' + label + '.txt', spikecount[label].count)
        
    numpy.savetxt(saving_dir+'/rate_' + 'L4exc' + '.txt', poprate['L4exc'].t)
    for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
        numpy.savetxt(saving_dir+'/rate_' + label + '.txt', poprate[label].rate)
        
############# end
plt.show()


## State monitor for voltages of simple spikes in a connected pair
## excitatory and then switched to a GABAergic neurons, most quiescent neurons



