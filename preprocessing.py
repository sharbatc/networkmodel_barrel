'''
    library for preprocessing the experimental data
    before using them in the simulation
    started December 2013 by Hesam SETAREH
    
    PSC : Exponential decay 
    PSP : Alpha shape (double exponential)

    Last modified : 11 Jun 2018 by Sharbat
'''

import numpy
import matplotlib.pyplot as plt
import time
import math
from brian2 import *    
    

def Neuron_parameters():
    
    El = 0*mV
    tau_m = 37.4*ms
    tau_syn = 7.4*ms
    R = 210*Mohm
    
    eqs = '''
    dv/dt = (-(v-El) + R*I)/tau_m : volt
    I_s : amp
    I = I_e + I_i + I_s : amp
    dI_e/dt = -I_e/tau_syn : amp
    dI_i/dt = -I_i/tau_syn : amp
    '''
    
    return eqs

def find_nearest(array, value):
    '''
    find index of nearest elements of value in array
    '''
    
    return numpy.argmin(numpy.abs(array-value))

def PSP_alpha(R,w,tau_syn,tau_m):
    '''
    Function for plotting and analysis the PSP caused by Exp-decay PSC
    R        : Membrane Resistance - MOhm
    w        : Simulation weight (Peak of exponential PSC) - nA
    tau_syn  : Synaptic time constant - ms
    tau_m    : Membrane time constant of postsynaptic neuron - ms
    
    calculated data:
    peak      = Weights of the synapse in the experimental terminology
    peak_time = Time of the peak
    rise_time = time from 0.2peak to 0.8peak
    slope     = (0.8peak - 0.2peak)/rise_time
    half_width= time from 0.5peak to 0.5 peak
    '''
    
    t = numpy.arange(0,50*tau_syn,0.1)
    PSP = (1.0*R*w*tau_syn)/(1.0*tau_m-tau_syn)*(numpy.exp(-t/tau_m)-numpy.exp(-t/tau_syn))
    
    peak = numpy.max(PSP)
    peak_time = numpy.argmax(PSP)/10.0
    
    peak0_2_time = find_nearest(PSP[:peak_time*10], 0.2*peak)/10.0
    peak0_8_time = find_nearest(PSP[:peak_time*10], 0.8*peak)/10.0
    
    rise_time = peak0_8_time - peak0_2_time
    PSP_slope = (0.8*peak - 0.2*peak)/(rise_time)
    
    peak0_5_before_time = find_nearest(PSP[:peak_time*10], 0.5*peak)/10.0
    peak0_5_after_time = find_nearest(PSP[peak_time*10:], 0.5*peak)/10.0 + peak_time
    
    half_width = peak0_5_after_time - peak0_5_before_time    
    
    print('peak= ' + str(peak) + '\trise_time= ' + str(rise_time) + '\thalf-width= ' + str(half_width))
    
    plt.plot(t,PSP)
    plt.grid()
    plt.show()    


def Compute_halfwidth(tau_syn,tau_m):
    
    '''
    Computing PSP half width from tau_syn and tau_m
    '''
    
    t = numpy.arange(0,50*tau_m,0.1)
    double_exp = numpy.exp(-t/tau_m)-numpy.exp(-t/tau_syn)
    peak_time = numpy.argmax(double_exp)/10.0
    double_exp_peak = (1.0*tau_syn/tau_m)**(1.0*tau_syn/(tau_m-tau_syn)) - (1.0*tau_syn/tau_m)**(1.0*tau_m/(tau_m-tau_syn))
    
    '''
    if(tau_syn>1.0):
        print tau_syn
        myabs = numpy.abs(double_exp-0.5*double_exp_peak)
        #plt.plot(t,myabs)
        plt.plot(t,double_exp)
        plt.grid()
        plt.show()
    '''
    
    if peak_time==0.0:
        return 0

    t1 = find_nearest(double_exp[:int(peak_time*10)], 0.5*double_exp_peak)/10.0
    t2 = find_nearest(double_exp[int(peak_time*10):], 0.5*double_exp_peak)/10.0 + peak_time
    
    
    
    return int(t2-t1)
    
def Find_tau_syn(half_width,tau_m):
    '''
    Find tau_syn based on tau_m and half_width
    '''
    
    t = numpy.arange(0,20,0.1)
    myhw = numpy.zeros([len(t)])
    
    for i in t:
        if i==0.0:
            continue

        myhw[int(i*10)] = Compute_halfwidth(t[int(i*10)], tau_m)
        
    optimal_t = numpy.argmin(numpy.abs(myhw - half_width))
    
    return optimal_t/10.0


def Find_w(peak, tau_syn, tau_m, R):
    '''
    Find w_simulation based on w_experiment (peak)
    R in MOhm
    peak in mV
    return w in nano ampere
    '''
    
    double_exp_peak = (1.0*tau_syn/tau_m)**(1.0*tau_syn/(tau_m-tau_syn)) - (1.0*tau_syn/tau_m)**(1.0*tau_m/(tau_m-tau_syn))
    return (peak*(tau_m-tau_syn))/(R*tau_syn*double_exp_peak)
    

def Fit_PSP(peak, half_width, tau_m, R):
    '''
    NOTE (Sharbat) : Use half-width means here from Avermann
    
    Fits alpha shape PSP (Exp decay PSC) using experimental data
    try to produce PSP pulse with same peak and half_width of experimental PSP
    peak    : experimental weight in mV
    half_width : experimental half width of PSP in ms
    tau_m   : membrane potential time constant in ms
    R       : membrane resistance in MOhm   
    
    calculated data
    tau_syn    : synaptic time constant in ms
    w          : weight of simulation (peak of exponential decay PSC) in nano ampere
    '''
    
    tau_syn = Find_tau_syn(half_width, tau_m)
    w = Find_w(peak, tau_syn, tau_m, R)
    
    return tau_syn, w
    

def CUBA_to_COBA(w_sim_cuba, El, E_syn):
    '''
    convert synaptic weight in current based framework
    to synaptic weight in conductance based framework
     w_sim_cuba: simulation weight of current based (nA)
                 WITH SIGN (positiv for exc. syn. and 
                 negative for inh. syn.)
    El: resting potential of neuron (mV)
    E_syn: Reversal potential of synapse (mV)
    returns synaptic weight in cond. based framework (micro siemens)
    '''
    
    return - w_sim_cuba / (El - E_syn)
