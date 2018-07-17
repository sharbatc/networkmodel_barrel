import brian2 as b2
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from input_factory import get_step_current
from plot_tools import plot_voltage_and_current_traces

model_eqs = '''
dv/dt = (-gL*(v-v_rest) - w1 - w2 + I_ext(t,i))/C : volt

dw1/dt = -w1/tau_w1 : amp
dw2/dt = -w2/tau_w2 : amp

dvt1/dt = -vt1/tau_vt1 : volt
dvt2/dt = -vt2/tau_vt2 : volt

vt = v_thresh + vt1 + vt2 : volt
lambda_t = lambda_0*exp((v-vt)/del_v): Hz
'''
reset_eqs = '''
v = v_reset
w1+=amp_w1
w2+=amp_w2
vt1+=amp_vt1
vt2+=amp_vt2
'''

exc_df = pd.read_csv('data/exc.txt',header = None)
exc_df.columns =['C (nF)','gL (usiemens)','v_rest (mV)','v_reset (mV)','tau_refr (ms)',\
             'v_thresh (mV)','del_v (mV)', 'amp_w1 (nA)','tau_w1 (ms)','amp_w2 (nA)',\
             'tau_w2 (ms)','amp_vt1 (mV)','tau_vt1 (ms)','amp_vt2 (mV)','tau_vt2 (ms)']


exc_param = exc_df.iloc[1,:] #set which value to try out, set any if running through all

lambda_0 = 10 * b2.kHz

C = exc_param['C (nF)'] * b2.nF
gL = exc_param['gL (usiemens)'] * b2.usiemens
v_rest = exc_param['v_rest (mV)'] * b2.mV
v_reset = exc_param['v_reset (mV)'] * b2.mV
tau_refr = exc_param['tau_refr (ms)'] * b2.ms

v_thresh = exc_param['v_thresh (mV)'] * b2.mV
# del_v = exc_param['del_v (mV)'] * b2.mV
del_v = 0.1 * b2.mV
amp_w1 = exc_param['amp_w1 (nA)'] * b2.nA
tau_w1 = exc_param['tau_w1 (ms)'] * b2.ms
amp_w2 = exc_param['amp_w2 (nA)'] * b2.nA
tau_w2 = exc_param['tau_w2 (ms)'] * b2.ms

amp_vt1 = exc_param['amp_vt1 (mV)'] * b2.mV
tau_vt1 = exc_param['tau_vt1 (ms)'] * b2.ms
amp_vt2 = exc_param['amp_vt2 (mV)'] * b2.mV
tau_vt2 = exc_param['tau_vt2 (ms)'] * b2.ms


## amplitude to be played with 
I_ext = get_step_current(t_start = 25, t_end = 125, unit_time = 1*b2.ms, amplitude = 0.25*b2.namp)
time = 150 * b2.ms

EXC = b2.NeuronGroup(1, model = model_eqs, reset = reset_eqs, threshold = "rand() < lambda_t*dt",refractory = tau_refr, method = 'rk4')
EXC.v = v_rest
EXC.vt1 = 0 * b2.mV
EXC.vt2 = 0 * b2.mV
EXC.w1 = EXC.w2 = 0*b2.nA

voltage_monitor = b2.StateMonitor(EXC, ['v','vt','lambda_t'], record=True)
spike_monitor = b2.SpikeMonitor(EXC, variables = ["v"])

print("Before simulation" , datetime.datetime.now())

b2.run(time)

print("After simulation" , datetime.datetime.now())

fig = plt.figure()
plot_voltage_and_current_traces(voltage_monitor, spike_monitor, I_ext)
plt.show()