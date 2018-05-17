'''
	Thesis Project
	Testing single neuron models

	Started : 13 May 2018 by Sharbat
	Last modified : 17 May 2018 by Sharbat
'''
from brian2 import *
import numpy
import matplotlib.pyplot as plt
import math


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

reset_1='''
v=v_reset_1
w1+=amp_w1_1
w2+=amp_w2_1
vt1+=amp_vt1_1
vt2+=amp_vt2_1
'''