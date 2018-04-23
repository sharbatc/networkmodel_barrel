'''
    Pozzorinin et al (PLOS CB, 2015) : https://doi.org/10.1371/journal.pcbi.1004275
    Simulator: Brian2
    Testing different exc. neuron parameters sets of Skander's model
    Started : 15 April, 2015 by Hesam SETAREH
    Last modified : 23 April, 2018 by Sharbat
'''

import numpy
import matplotlib.pyplot as plt
from brian2 import *
import datetime
import os
import pickle

# custom functions 
import preprocessing
import analysis

# might be needed later

def step_current(I_amp, I_tstart, I_tend, end_time):
    tend = end_time
    Step_current = numpy.zeros(tend)
    for t in range(tend):
        if(I_tstart <= t and t <= I_tend):
            Step_current[t] = I_amp
    
    return Step_current

def ramp_current(I_ampmax, I_tstart, I_tend, end_time):
	tend = end_time
	Ramp_current = numpy.zeros(tend)
	for t in range(tend):
		if(I_tstart <= t and t <= I_tend):
			Ramp_current[t] = Ramp_current[t-1] + I_ampmax/(I_tend - I_tstart)

	return Ramp_current

## params
defaultclock.dt = 0.1*ms
time = 5000
param_set_exc = 2
param_set_inh = 0

type = 'fs' # it could be 'exc', 'fs' or 'nfs' - to be asked about current nom
			# currently, nfs is inh

exc_param = numpy.loadtxt('data/' + type + '.txt', delimiter = ',')[param_set_exc]
inh_param = numpy.loadtxt('data/nfs.txt', delimiter = ',')[param_set_inh]

tau_exc_exc, w_exc_exc = preprocessing.Fit_PSP(1.56, 47, exc_param[0]/exc_param[1], 1/exc_param[1])
print('tau_exc_exc: ' + str(tau_exc_exc) + "\tw_exc_exc: " + str(w_exc_exc))

tau_exc_inh, w_exc_inh = preprocessing.Fit_PSP(0.82, 47, inh_param[0]/inh_param[1], 1/inh_param[1])
print('tau_exc_inh: ' + str(tau_exc_inh) + "\tw_exc_inh: " + str(w_exc_inh))

tau_inh_exc, w_inh_exc = preprocessing.Fit_PSP(0.52, 20, exc_param[0]/exc_param[1], 1/exc_param[1])
print('tau_inh_exc: ' + str(tau_inh_exc) + "\tw_inh_exc: " + str(w_inh_exc))

tau_inh_inh, w_inh_inh = preprocessing.Fit_PSP(0.56, 20, inh_param[0]/inh_param[1], 1/inh_param[1])
print('tau_ihn_inh: ' + str(tau_inh_inh) + "\tw_inh_inh: " + str(w_inh_inh))

tau_exc_nh, w_exc_nh = preprocessing.Fit_PSP(0.37, 47, exc_param[0]/exc_param[1], 1/exc_param[1])
print('tau_exc_nh: ' + str(tau_exc_nh) + "\tw_exc_nh: " + str(w_exc_nh))

#################### Excitatory Neuron Model

rate0_1 = defaultclock.dt/ms * Hz

tau_exc_1 = tau_exc_exc * ms
tau_inh_1 = tau_inh_exc * ms  

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
I_syn = I_exc + I_inh : amp
dI_exc/dt = -I_exc/tau_exc_1 : amp
dI_inh/dt = -I_inh/tau_inh_1 : amp

C_1 :farad
gl_1 :siemens
El_1 :volt
v0_1 :volt
deltaV_1 :volt
tau_ref_1 :second
v_reset_1 :volt
rate0_1 :Hz
amp_w1_1 :amp
tau_w1_1 :second
amp_w2_1 :amp
tau_w2_1 :second


amp_vt1_1 :volt
tau_vt1_1 :second
amp_vt2_1 :volt
tau_vt2_1 :second

'''

reset_1='''
v=v_reset_1
w1+=amp_w1_1
w2+=amp_w2_1
vt1+=amp_vt1_1
vt2+=amp_vt2_1
'''

#################### Inhibitory Neuron Model

C_2 = inh_param[0] * nF
gl_2 = inh_param[1] * uS
El_2 = inh_param[2] * mV
v_reset_2 = inh_param[3] * mV
tau_ref_2 = inh_param[4] * ms
v0_2 = inh_param[5] * mV
deltaV_2 = inh_param[6] * mV

amp_w1_2 = inh_param[7] * nA
tau_w1_2 = inh_param[8] * ms
amp_w2_2 = inh_param[9] * nA
tau_w2_2 = inh_param[10] * ms


amp_vt1_2 = inh_param[11] * mV
tau_vt1_2 = inh_param[12] * ms
amp_vt2_2 = inh_param[13] * mV
tau_vt2_2 = inh_param[14] * ms

rate0_2 = defaultclock.dt/ms * Hz

tau_exc_2 = tau_exc_inh * ms
tau_inh_2 = tau_inh_inh * ms  

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
I_syn = I_exc + I_inh : amp
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

######################### Creating populations
EXC = []

##just to make them fire
for i in range(0,7):
    exc_param = numpy.loadtxt('data/' + type + '.txt', delimiter = ',')[i]
    tmp = NeuronGroup(1, model=eqs_1, reset = reset_1, 
                      threshold ='rand()<rate*dt', refractory = exc_param[4]*ms)
    EXC.append(tmp)
    EXC[i].C_1 = exc_param[0] * nF
    EXC[i].gl_1 = exc_param[1] * uS
    EXC[i].El_1 = exc_param[2] * mV
    EXC[i].v_reset_1 = exc_param[3] * mV
    EXC[i].tau_ref_1 = exc_param[4] * ms
    EXC[i].v0_1 = exc_param[5] * mV
    EXC[i].deltaV_1 = exc_param[6] * mV

    EXC[i].amp_w1_1 = exc_param[7] * nA
    EXC[i].tau_w1_1 = exc_param[8] * ms
    EXC[i].amp_w2_1 = exc_param[9] * nA
    EXC[i].tau_w2_1 = exc_param[10] * ms

    EXC[i].amp_vt1_1 = exc_param[11] * mV
    EXC[i].tau_vt1_1 = exc_param[12] * ms
    EXC[i].amp_vt2_1 = exc_param[13] * mV
    EXC[i].tau_vt2_1 = exc_param[14] * ms

    EXC[i].rate0_1 = defaultclock.dt/ms * Hz

    EXC[i].v = EXC[i].El_1

for i in range(0,7):
    print(EXC[i].C_1/nF)

########################### Stimulus
ctime, my_step_current = analysis.gen_ou()
my_step_current *= nA/40

#my_step_current = step_current(0.324 * nA, 50, 250, 400)
#my_step_current += numpy.random.rand(400) * nA / 10

coeff = numpy.zeros([7])
coeff[0] = 1.75
coeff[1] = 1.8
coeff[2] = 1.1
coeff[3] = 1.7
coeff[4] = 1.5
coeff[5] = 0.92
coeff[6] = 1.9

## check this
for i in range(0,7):
    EXC[i].I_ext = TimedArray(my_step_current,dt=1*ms)/coeff[i]

########################### Recorders
V_EXC = []
for i in range(0,7):
    tmp = StateMonitor(EXC[i], 'v', record = True)
    V_EXC.append(tmp)
    
SPIKE_EXC = []
for i in range(0,7):
    tmp = SpikeMonitor(EXC[i])
    SPIKE_EXC.append(tmp)

############################## Run
print("Before simulation" , datetime.datetime.now())

run(time * ms)

print("After simulation" , datetime.datetime.now())


############################# Showing results

AP_shape = numpy.array([0])    

figure()
title(type + ' voltage')
for i in range(0,7):
    subplot(7,1,i+1)
    plot(V_EXC[i].times/ms,analysis.replace_AP(V_EXC[i][0]/mV,SPIKE_EXC[i].spiketimes[0],AP_shape), lw=2)
    yticks([-60, -30, 0], [-60, -30, 0])

figure()
title('current')
plot(ctime, my_step_current, lw=2)
xlabel('time (ms)')
ylabel('current (nA)')
xlim(0, time)

print("The end" , datetime.datetime.now())
show()
