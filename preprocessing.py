'''
    library for preprocessing the experimental data
    before using them in the simulation
    started December 2013 by Hesam SETAREH
    
    PSC : Exponential decay 
    PSP : Alpha shape (double exponential)
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

def find_2elements_lognormal(p_hat_1,p_hat_2,lognormal_mean,lognormal_std):
    '''
    fit a 2 elements distribution (w_1,w_2) with distribution (p_hat1,p_hat2)
    based on a lognormal distribution parameters
    1 ==> weak
    2 ==> strong
    return (w_1, w_2)
    '''
    
    if numpy.abs(p_hat_1+p_hat_2-1) > 0.05:
        raise NameError("p1+p2 should equal 1")
    
    x = numpy.arange(0,5,0.01)
    cdf = numpy.arange(0,5,0.01)
    pdf = numpy.arange(0,5,0.01)
    for i in range(1,x.__len__()):
        cdf[i] =  0.5 + 0.5*math.erf((numpy.log(x[i])-lognormal_mean)/(numpy.sqrt(2)*lognormal_std))
        pdf[i] = 1/(x[i]*lognormal_std*numpy.sqrt(2*math.pi)) * numpy.exp(-(numpy.log(x[i])-lognormal_mean)**2/(2*lognormal_std**2))
        
    index_star = find_nearest(cdf, p_hat_1)
    w_star = x[index_star]
    #print w_star

    w_pdf = x*pdf # w*p(w)
     
    w_1 = numpy.sum(w_pdf[:index_star])/(p_hat_1*100)   # 100 since dx = 0.01
    w_2 = numpy.sum(w_pdf[index_star:])/(p_hat_2*100)   # 100 since dx = 0.01
    
    return w_1, w_2
    
def calculate_2elements(p_ave,p2,N_h,N_nh,lognormal_mean, lognormal_std):
    '''
    calculate 2elements given p_[hub to hub]
    based on lognormal distribution parameters
    only hub->hub connections are dense, others are sparse
    
    p2 = p_[hub to hub]
    p_ave = p_average
    return (p1, p2, w_1, w_2)
    '''
    
    N = N_h + N_nh
    p1 = (N**2*p_ave - N_h**2*p2)/(N_nh**2 + 2*N_h*N_nh)
    
    p_hat_2 = (N_h**2*p2 + N_h*N_nh*p1)/(N**2*p_ave)
    p_hat_1 = ((N_nh**2 + N_h*N_nh)*p1)/(N**2*p_ave)
    
    #print p_hat_1, p_hat_2, p_hat_1+p_hat_2
    
    w_1, w_2 = find_2elements_lognormal(p_hat_1, p_hat_2, lognormal_mean, lognormal_std)

    #print w_1*p_hat_1 + w_2*p_hat_2

    return p1, p2, w_1, w_2
    

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


#print Fit_PSP(0.48, 30, 15, 150)

# L5A approx: tau_m = 37.6, R = 210 Mohm, peak = 0.66, rise_time = 2.99, half-width = 34
 
#PSP_alpha(210,0.062,2.3,37.6)

#PSP_alpha(1,1,10,20)

#log_mean, log_std = -0.84616167337581372, 0.92454423592588297
#print calculate_2elements(p_ave=0.19,p2=0.5,N_h=95,N_nh=454-95,lognormal_mean=log_mean, lognormal_std=log_std)

