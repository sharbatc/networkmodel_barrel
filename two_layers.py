'''
    Project Disinhibition
    NEW VERSION (new pop. sizes and new network regulation)
    Implementation of L4 and L2/3 using GIF
    L4 contains exc, pv(fs) and stt, but not vip neurons
    L2/3 contains exc, pv(fs) and stt and vip neurons
    Hesam SETAREH, 4 July 2017
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
param_set_fs  = params.param_set_fs
param_set_sst = params.param_set_sst
param_set_vip = params.param_set_nfs


exc_param = numpy.loadtxt('data/exc.txt', delimiter = ',')[param_set_exc]
fs_param  = numpy.loadtxt('data/fs.txt' , delimiter = ',')[param_set_fs]
sst_param  = numpy.loadtxt('data/nfs.txt' , delimiter = ',')[param_set_sst]
vip_param  = numpy.loadtxt('data/nfs.txt' , delimiter = ',')[param_set_vip]


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


# w = eta
# vt= gamma

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

# w = eta
# vt= gamma

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

rate0_3 = defaultclock.dt/ms * Hz * 1000

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
pops = {}

pops['L4exc'] = NeuronGroup(params.size['L4']['exc'], model=eqs_1, reset = reset_1, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_1)
pops['L4pv'] = NeuronGroup(params.size['L4']['pv'], model=eqs_2, reset = reset_2, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_2)
pops['L4sst'] = NeuronGroup(params.size['L4']['sst'], model=eqs_3, reset = reset_3, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_3)

pops['L23exc'] = NeuronGroup(params.size['L23']['exc'], model=eqs_1, reset = reset_1, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_1)
pops['L23pv'] = NeuronGroup(params.size['L23']['pv'], model=eqs_2, reset = reset_2, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_2)
pops['L23sst'] = NeuronGroup(params.size['L23']['sst'], model=eqs_3, reset = reset_3, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_3)
pops['L23vip'] = NeuronGroup(params.size['L23']['vip'], model=eqs_4, reset = reset_4, threshold = PoissonThreshold(state='lambda'), refractory=tau_ref_4)


#thalamus = PoissonGroup(600, rates = lambda t:(1+math.sin(t*5))*7*Hz)

#thalamus = PoissonGroup(600, rates = lambda t:heavyside(math.sin(t*5))*9*Hz)

#thalamus = PoissonGroup(600, rates = lambda t:(step(t-1000*ms)-step(t-2000*ms))*15*Hz) 
thalamus = PoissonGroup(600, rates = lambda t:(multistep(t/second))*15*Hz) 


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
conn_L4exc_L4exc = Synapses(pops['L4exc'], pops['L4exc'], state='I_exc', sparseness=params.conn_param['L4_L4']['exc_exc']['p'], weight = params.conn_param['L4_L4']['exc_exc']['w']*nA)
conn_L4pv_L4pv = Synapses(pops['L4pv'], pops['L4pv'], state='I_inh', sparseness=params.conn_param['L4_L4']['pv_pv']['p'], weight =1.1*-params.conn_param['L4_L4']['pv_pv']['w']*nA) ##### w = 1.3
conn_L4sst_L4sst = Synapses(pops['L4sst'], pops['L4sst'], state='I_inh', sparseness=params.conn_param['L4_L4']['sst_sst']['p'], weight = 0.8*-params.conn_param['L4_L4']['sst_sst']['w']*nA)

#miu, sigma, _ = connectivity.normal_to_lognormal(params.conn_param['L4_L4']['exc_pv']['w'], params.conn_param['L4_L4']['exc_pv']['w']/2)
#w = connectivity.lognormal_weight(params.size['L4']['exc'], params.size['L4']['pv'], miu, sigma, self_conn = True)
#w = numpy.multiply(w, connectivity.Synapses_matrix_random(params.size['L4']['exc'], params.size['L4']['pv'], params.conn_param['L4_L4']['exc_pv']['p'], self_conn=True))
#conn_L4exc_L4pv = Synapses(L4_exc, L4_pv, state='I_exc', weight = w*nA)
conn_L4exc_L4pv = Synapses(pops['L4exc'], pops['L4pv'], state='I_exc', sparseness=params.conn_param['L4_L4']['exc_pv']['p'], weight = 0.6*params.conn_param['L4_L4']['exc_pv']['w']*nA)

#miu, sigma, _ = connectivity.normal_to_lognormal(params.conn_param['L4_L4']['pv_exc']['w'], params.conn_param['L4_L4']['pv_exc']['w']/2)
#w = connectivity.lognormal_weight(params.size['L4']['pv'], params.size['L4']['exc'], miu, sigma, self_conn = True)
#w = numpy.multiply(w, connectivity.Synapses_matrix_random(params.size['L4']['pv'], params.size['L4']['exc'], params.conn_param['L4_L4']['pv_exc']['p'], self_conn=True))
#conn_L4pv_L4exc = Synapses(L4_pv, L4_exc, state='I_inh', weight = -w*nA)
conn_L4pv_L4exc = Synapses(pops['L4pv'], pops['L4exc'], state='I_inh', sparseness=params.conn_param['L4_L4']['pv_exc']['p'], weight = 0.7*-params.conn_param['L4_L4']['pv_exc']['w']*nA)

conn_L4exc_L4sst = Synapses(pops['L4exc'], pops['L4sst'], state='I_exc', sparseness=params.conn_param['L4_L4']['exc_sst']['p'], weight = 1.15*params.conn_param['L4_L4']['exc_sst']['w']*nA) # weight = *1.1
#mystp=STP(conn_L4exc_L4sst, taud=20*ms,tauf=1000*ms, U=0.05) # for parameters look at  krishnamurthy 2012 plos one
conn_L4sst_L4exc = Synapses(pops['L4sst'], pops['L4exc'], state='I_inh', sparseness=params.conn_param['L4_L4']['sst_exc']['p'], weight =-params.conn_param['L4_L4']['sst_exc']['w']*nA)

conn_L4sst_L4pv = Synapses(pops['L4sst'], pops['L4pv'], state='I_inh', sparseness=params.conn_param['L4_L4']['sst_pv']['p'], weight =-params.conn_param['L4_L4']['sst_pv']['w']*nA)


####### inside L2/3
conn_L23exc_L23exc = Synapses(pops['L23exc'], pops['L23exc'], state='I_exc', sparseness=params.conn_param['L23_L23']['exc_exc']['p'], weight = params.conn_param['L23_L23']['exc_exc']['w']*nA)
conn_L23pv_L23pv = Synapses(pops['L23pv'], pops['L23pv'], state='I_inh', sparseness=params.conn_param['L23_L23']['pv_pv']['p'], weight =-params.conn_param['L23_L23']['pv_pv']['w']*nA)
conn_L23sst_L23sst = Synapses(pops['L23sst'], pops['L23sst'], state='I_inh', sparseness=params.conn_param['L23_L23']['sst_sst']['p'], weight = -params.conn_param['L23_L23']['sst_sst']['w']*nA)

conn_L23vip_L23vip = Synapses(pops['L23vip'], pops['L23vip'], state='I_inh', sparseness=params.conn_param['L23_L23']['vip_vip']['p'], weight = 0.7*-params.conn_param['L23_L23']['vip_vip']['w']*nA)

conn_L23exc_L23pv = Synapses(pops['L23exc'], pops['L23pv'], state='I_exc', sparseness=params.conn_param['L23_L23']['exc_pv']['p'], weight = 0.6*params.conn_param['L23_L23']['exc_pv']['w']*nA)

conn_L23pv_L23exc = Synapses(pops['L23pv'], pops['L23exc'], state='I_inh', sparseness=params.conn_param['L23_L23']['pv_exc']['p'], weight = 0.7*-params.conn_param['L23_L23']['pv_exc']['w']*nA)

conn_L23exc_L23sst = Synapses(pops['L23exc'], pops['L23sst'], state='I_exc', sparseness=params.conn_param['L23_L23']['exc_sst']['p'], weight = params.conn_param['L23_L23']['exc_sst']['w']*nA)
conn_L23exc_L23sst_stp = STP(conn_L23exc_L23sst, taud=20*ms,tauf=1000*ms, U=0.05) # for parameters look at  krishnamurthy 2012 plos one

conn_L23sst_L23exc = Synapses(pops['L23sst'], pops['L23exc'], state='I_inh', sparseness=params.conn_param['L23_L23']['sst_exc']['p'], weight =-params.conn_param['L23_L23']['sst_exc']['w']*nA)

#conn_L23sst_L23pv = Synapses(pops['L23sst'], pops['L23pv'], state='I_inh', sparseness=params.conn_param['L23_L23']['sst_pv']['p'], weight =-params.conn_param['L23_L23']['sst_pv']['w']*nA)

conn_L23exc_L23vip = Synapses(pops['L23exc'], pops['L23vip'], state='I_exc', sparseness=params.conn_param['L23_L23']['exc_vip']['p'], weight = 1.3*params.conn_param['L23_L23']['exc_vip']['w']*nA)
conn_L23vip_L23exc = Synapses(pops['L23vip'], pops['L23exc'], state='I_inh', sparseness=params.conn_param['L23_L23']['vip_exc']['p'], weight =-1.2*params.conn_param['L23_L23']['vip_exc']['w']*nA)
conn_L23vip_L23sst = Synapses(pops['L23vip'], pops['L23sst'], state='I_inh', sparseness= 1.4*params.conn_param['L23_L23']['vip_sst']['p'], weight =-1.5*params.conn_param['L23_L23']['vip_sst']['w']*nA)
conn_L23vip_L23pv = Synapses(pops['L23vip'], pops['L23pv'], state='I_inh', sparseness=params.conn_param['L23_L23']['vip_pv']['p'], weight =-params.conn_param['L23_L23']['vip_pv']['w']*nA)


####### L4 to L2/3
conn_L4exc_L23exc = Synapses(pops['L4exc'], pops['L23exc'], state='I_exc', sparseness=params.conn_param['L4_L23']['exc_exc']['p'], weight = 0.6*params.conn_param['L4_L23']['exc_exc']['w']*nA)

conn_L4exc_L23pv = Synapses(pops['L4exc'], pops['L23pv'], state='I_exc', sparseness=params.conn_param['L4_L23']['exc_pv']['p'], weight = 0.8*params.conn_param['L4_L23']['exc_pv']['w']*nA)
conn_L4exc_L23vip = Synapses(pops['L4exc'], pops['L23vip'], state='I_exc', sparseness=params.conn_param['L4_L23']['exc_pv']['p'], weight = params.conn_param['L4_L23']['exc_pv']['w']*nA)



####### from Thalamus
conn_th_L4exc = Synapses(thalamus, pops['L4exc'], sparseness=0.2, weight = 0.02*nA, state='I_exc')  # w should be fixed
conn_th_L4pv = Synapses(thalamus, pops['L4pv'], sparseness=0.10, weight = 0.01*nA, state='I_exc')  # w should be fixed

conn_th_L23exc = Synapses(thalamus, pops['L23exc'], sparseness=0.05, weight = 0.02*nA, state='I_exc')  # w should be fixed
conn_th_L23pv = Synapses(thalamus, pops['L23pv'], sparseness=0.05, weight = 0.01*nA, state='I_exc')  # w should be fixed


###### noise

noise_L4exc  = PoissonInput(pops['L4exc'], 600, rate=10*Hz, weight=0.015*nA, state='I_noise')  # poisson input to L4_exc # 600, 10
noise_L4pv = PoissonInput(pops['L4pv'], 500, rate=5*Hz, weight=0.015*nA, state='I_noise')    # poisson input to L4_pv # 400, 5
noise_L4sst = PoissonInput(pops['L4sst'], 250, rate=5*Hz, weight=0.015*nA, state='I_noise')   # poisson input to L4_sst


noise_L23exc  = PoissonInput(pops['L23exc'], 600, rate=10*Hz, weight=0.01*nA, state='I_noise')  # poisson input to L4_exc # 600, 10
noise_L23pv = PoissonInput(pops['L23pv'], 500, rate=5*Hz, weight=0.015*nA, state='I_noise')    # poisson input to L4_pv # 400, 5
noise_L23sst = PoissonInput(pops['L23sst'], 250, rate=5*Hz, weight=0.015*nA, state='I_noise')   # poisson input to L4_sst
noise_L23vip = PoissonInput(pops['L23vip'], 100, rate=5*Hz, weight=0.015*nA, state='I_noise')    # poisson input to L4_pv # 400, 5


############## recorders
rec_I_noise = {}
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

rec_I_noise['L23vip'] = StateMonitor(pops['L23vip'], 'I_noise', record = True)
rec_I_exc['L23vip'] = StateMonitor(pops['L23vip'], 'I_exc', record = True)
rec_I_inh['L23vip'] = StateMonitor(pops['L23vip'], 'I_inh', record = True)


rec_I_noise['L23sst'] = StateMonitor(pops['L23sst'], 'I_noise', record = True)
rec_I_exc['L23sst'] = StateMonitor(pops['L23sst'], 'I_exc', record = True)
rec_I_inh['L23sst'] = StateMonitor(pops['L23sst'], 'I_inh', record = True)

rec_I_noise['L4sst'] = StateMonitor(pops['L4sst'], 'I_noise', record = True)
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

spikecount['L4exc'] = SpikeCounter(pops['L4exc'])
spikecount['L4sst'] = SpikeCounter(pops['L4sst'])
spikecount['L4pv'] = SpikeCounter(pops['L4pv'])

spikecount['L23exc'] = SpikeCounter(pops['L23exc'])
spikecount['L23sst'] = SpikeCounter(pops['L23sst'])
spikecount['L23pv'] = SpikeCounter(pops['L23pv'])
spikecount['L23vip'] = SpikeCounter(pops['L23vip'])


spikes['thalamus'] = SpikeMonitor(thalamus)
spikes['L4exc'] = SpikeMonitor(pops['L4exc'])
spikes['L4pv'] = SpikeMonitor(pops['L4pv'])
spikes['L4sst'] = SpikeMonitor(pops['L4sst'])

spikes['L23exc'] = SpikeMonitor(pops['L23exc'])
spikes['L23pv'] = SpikeMonitor(pops['L23pv'])
spikes['L23sst'] = SpikeMonitor(pops['L23sst'])
spikes['L23vip'] = SpikeMonitor(pops['L23vip'])


poprate['thalamus'] = PopulationRateMonitor(thalamus, bin=20*ms)
poprate['L4exc'] = PopulationRateMonitor(pops['L4exc'], bin=20*ms)
poprate['L4pv'] = PopulationRateMonitor(pops['L4pv'], bin=20*ms)
poprate['L4sst'] = PopulationRateMonitor(pops['L4sst'], bin=20*ms)

poprate['L23exc'] = PopulationRateMonitor(pops['L23exc'], bin=20*ms)
poprate['L23pv'] = PopulationRateMonitor(pops['L23pv'], bin=20*ms)
poprate['L23sst'] = PopulationRateMonitor(pops['L23sst'], bin=20*ms)
poprate['L23vip'] = PopulationRateMonitor(pops['L23vip'], bin=20*ms)

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
    mean_I_noise = numpy.mean(rec_I_noise[label].values, axis=0)/nA
    mean_I_exc = numpy.mean(rec_I_exc[label].values, axis=0)/nA
    mean_I_inh = numpy.mean(rec_I_inh[label].values, axis=0)/nA
    mean_I_total = mean_I_exc + mean_I_inh
    plt.plot(rec_I_exc[label].times/ms, mean_I_noise, 'green', lw=2, label='noise')
    plt.plot(rec_I_exc[label].times/ms, mean_I_exc, 'blue', lw=2, label='Exc')
    plt.plot(rec_I_exc[label].times/ms, mean_I_inh, 'red', lw=2, label='Inh')
    plt.plot(rec_I_exc[label].times/ms, mean_I_total, 'black', lw=2, label='Total')
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
        plt.plot(rec_v[label].times/ms, v0, 'blue', lw=2)
        plt.subplot(212)
        plt.plot(rec_v[label].times/ms, v1, 'red', lw=2)
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
plt.figure()
raster_plot(spikes['L4sst'])
plt.title('L4sst - raster')


plt.figure()
raster_plot(spikes['L23exc'])
plt.title('L23exc - raster')

plt.figure()
raster_plot(spikes['L23pv'])
plt.title('L23pv - raster')

plt.figure()
raster_plot(spikes['L23sst'])
plt.title('L23sst - raster')

plt.figure()
raster_plot(spikes['L23vip'])
plt.title('L23vip - raster')


############ not correct
#print "\nRates of L4:"
#for label in ['exc', 'pv', 'sst']:
#    print label + ' rate, before: ' + str(numpy.mean(poprate['L4'+label].rate[10:50])) + ', after: ' + str(numpy.mean(poprate['L4'+label].rate[50:])) 

#print "\nRates of L2/3:"
#for label in ['exc', 'pv', 'vip', 'sst']:
#    print label + ' rate, before: ' + str(numpy.mean(poprate['L23'+label].rate[10:50])) + ', after: ' + str(numpy.mean(poprate['L23'+label].rate[50:])) 


plt.figure()
plt.plot(poprate['L4exc'].times/ms, poprate['L4exc'].rate, label='exc', lw=2)
plt.plot(poprate['L4pv'].times/ms, poprate['L4pv'].rate, label='pv', lw=2)
plt.plot(poprate['L4sst'].times/ms, poprate['L4sst'].rate, label='sst', lw=2)
plt.legend()
plt.grid()
plt.title('L4 - firing rates')
plt.xlabel('time (ms)')
plt.ylabel('firing rate (Hz)')


plt.figure()
plt.plot(poprate['L23exc'].times/ms, poprate['L23exc'].rate, label='exc', lw=2)
plt.plot(poprate['L23pv'].times/ms, poprate['L23pv'].rate, label='pv', lw=2)
plt.plot(poprate['L23sst'].times/ms, poprate['L23sst'].rate, label='sst', lw=2)
plt.plot(poprate['L23vip'].times/ms, poprate['L23vip'].rate, label='vip', lw=2)
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
        numpy.save(saving_dir+'/voltages_times.txt', rec_v['L4exc'].times/ms)
        for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
            numpy.save(saving_dir+'/voltages_' + label + '.txt', rec_v[label].values/mV)
        
    for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
        numpy.savetxt(saving_dir+'/spikecount_' + label + '.txt', spikecount[label].count)
        
    numpy.savetxt(saving_dir+'/rate_' + 'L4exc' + '.txt', poprate['L4exc'].times)
    for label in ['L4exc', 'L4pv', 'L4sst', 'L23exc', 'L23pv', 'L23sst', 'L23vip']:
        numpy.savetxt(saving_dir+'/rate_' + label + '.txt', poprate[label].rate)
        
############# end
plt.show()






