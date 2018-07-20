# This code has been modified from that available at 
# the exercises for the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

# Started by Sharbat, 11 Jun 2018

import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy

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



def get_ou_current(I0, plot=False, unit_time=1*b2.ms, len_current=1000):
    tau = 10.0  # ms
    #Delta_T = 1.0/20  #ms, sampling freq = 20 kHz
    Delta_T = 1.0/10  #ms, sampling freq = 20 kHz
    #sigma = 1.0  # unitless
    sigma_0 = 1.0  # unitless
    Delta_sigma = 2.0  # unitless
    f = 0.2 * 0.001  # kHz 
    
    
    len = int(len_current / Delta_T)  # length of arrays: I and time
    I = numpy.zeros((len,1))  # nA
    time = numpy.arange(0, len_current, Delta_T)  # ms
    sigma = numpy.zeros(len)  # unitless
    
    I[0,0] = 0.0
    for counter in range(1, len):
        sigma[counter] = sigma_0 * (1 + Delta_sigma * numpy.sin(2*numpy.pi*f*time[counter]))
        I[counter] = I[counter-1] + (I0 - I[counter-1])/tau*Delta_T + numpy.sqrt(2*sigma[counter]**2*Delta_T/tau)*numpy.random.normal()  # N(0,1)
    
    if plot:
        plt.plot(time, I, lw=2)
    
        plt.plot(time, I0*(1-numpy.exp(-time/tau)), color='red', lw=2)
    
        plt.figure()
        plt.plot(time, sigma, color='green', lw=2)
        plt.grid()
    
        plt.show()

    I = I[:,0]/100 * b2.namp
    I = b2.TimedArray(I, dt = 1. * unit_time)

    return I