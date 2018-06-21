'''
	Thesis Project
	Testing single neuron models

	Started : 13 May 2018 by Sharbat
	Last modified : 4 Jun 2018 by Sharbat
'''
import brian2 as b2
import numpy
import matplotlib.pyplot as plt
import math
import params

param_set_exc = params.param_set_exc
param_set_nfs  = params.param_set_nfs
param_set_fs = params.param_set_fs

## tested params, might have to be changed ##

exc_param = numpy.loadtxt('data/exc.txt', delimiter = ',')[param_set_exc]

## get step current
def get_step_current(t_start, t_end, unit_time, amplitude, append_zero=True):

    """Creates a step current. If t_start == t_end, then a single
    entry in the values array is set to amplitude.
    Args:
        t_start (int): start of the step
        t_end (int): end of the step
        unit_time (Brian2 unit): unit of t_start and t_end. e.g. 0.1*brian2.ms
        amplitude (Quantity): amplitude of the step. e.g. 3.5*brian2.uamp
        append_zero (bool, optional): if true, 0Amp is appended at t_end+1.
        Without that trailing 0, Brian reads out the last value in the array (=amplitude) for all indices > t_end.
    Returns:
        TimedArray: Brian2.TimedArray
    """

    assert isinstance(t_start, int), "t_start_ms must be of type int"
    assert isinstance(t_end, int), "t_end must be of type int"
    assert b2.units.fundamentalunits.have_same_dimensions(amplitude, b2.amp), \
        "amplitude must have the dimension of current e.g. brian2.uamp"
    tmp_size = 1 + t_end  # +1 for t=0
    if append_zero:
        tmp_size += 1
    tmp = np.zeros((tmp_size, 1)) * b2.amp
    tmp[t_start: t_end + 1, 0] = amplitude
    curr = b2.TimedArray(tmp, dt=1. * unit_time)
    return curr

## exc neuron 

############### neural parameters

# code needs to be much better written
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

# tau_exc_1 = params.conn_param['L23_L23']['exc_exc']['tau_syn'] * ms
# tau_inh_1 = params.conn_param['L23_L23']['pv_exc']['tau_syn'] * ms


# w = eta
# vt= gamma

eqs_1 = '''
dv/dt = (-gl_1*(v-El_1)-w1-w2+0.324*nA)/C_1 : volt
dw1/dt = -w1/tau_w1_1 : amp
dw2/dt = -w2/tau_w2_1 : amp
dvt1/dt = -vt1/tau_vt1_1 : volt
dvt2/dt = -vt2/tau_vt2_1 : volt
vt = v0_1 + vt1 + vt2 : volt
lambda = lambda0_1*exp((v-vt)/deltaV_1): Hz
'''

reset_1='''
v=v_reset_1
w1+=amp_w1_1
w2+=amp_w2_1
vt1+=amp_vt1_1
vt2+=amp_vt2_1
'''

