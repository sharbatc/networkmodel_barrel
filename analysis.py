'''
    library for analyzing network dynamics
    should be used together with Brian simulator
    started November 2013 by Hesam SETAREH
    last modification: March 27, 2016
    last modification: comments by Sharbat, March 28, 2018
'''

import numpy
import matplotlib.pyplot as plt
import time
import scipy.integrate
import math
import matplotlib


## not sure where and why this is needed, starting here, to self-define functions? ###

def load_myfunc():
    '''
    loading negative values of function e^x^2 * (1+erf(x)) function from a file, interval = [0,99.9999], precision = 0.0001
    '''
    global myfunc_table 
    myfunc_table = numpy.loadtxt('myfunc_part1.txt')
    return myfunc_table

def load_myfunc2():
    '''
    loading negative values of function e^x^2 * (1+erf(x)) function from a file, interval = [100,199.9999], precision = 0.0001
    '''
    global myfunc_table2 
    myfunc_table2 = numpy.loadtxt('myfunc_part2.txt')
    return myfunc_table2

### ends here ###

def find_fixed_points(I,rate1, rate2):  # should be replaced with same function in analysis(Sem4)
    '''
    find positions of two fixed points (unstable and stable) in two-relations system
    should be completed to find the first fixed point (off point) in case of non-zero
    '''
    dif = rate1 - rate2
    positive = numpy.nonzero(dif>0)[0]
    len_positive = positive.__len__()
    
    
    arr = numpy.nonzero(numpy.diff(positive)>1)[0]
    
    if arr.__len__()!=0:
        start = arr[0]+1
    else:
        start = positive[0]
    
    
    if(len_positive<=1):
        return 0, 0, 0, 0
    
    return I[positive[start]], rate1[positive[start]], I[positive[len_positive-1]], rate1[positive[len_positive-1]]


def rI_network_multiple(params,I_max):
    '''
    computing second relation between rate and I_syn
    based of network activity
    for multiple populations (population 0 is the host)
    '''
    I = numpy.arange(0,I_max,0.001) # nA
    
    rate = I/(params[0].N*params[0].p*params[0].tau_syn*params[0].w*1.25)  ################################ 1.25 should be removed
    
    for i in range(1,params.__len__()):
        rate -= (0.001*params[i].N*params[i].p*params[i].tau_syn*params[i].w*params[i].rate)/(params[0].N*params[0].p*params[0].tau_syn*params[0].w)
    
    return I, 1000*rate
    

def rI_network(N, p, w, tau_syn, I_max):
    '''
    computing second relation between rate and I_syn
    based of network activity
    N          : number of neurons
    p          : connection probability
    w          : simulation weight (nA)
    tau_syn    : synaptic time constant (ms)
    I_max      : maximum synaptic current (nA)
    q is equal to tau_syn
    '''
    I = numpy.arange(0,I_max,0.001) # nA
    
    rate = I/(N*p*tau_syn*w)
    
    return I, 1000*rate
    
    

def sigma(R, tau_m, tau_syn, N, p, w, r):
    '''
    calculating sigma for noisy gain function
    var = 2*var_voltage
    R       : resistance (Mohm)
    tau_m   : membrane time constant (ms)
    tau_syn : synaptic time constant (ms)
    N       : number of neurons
    p       : connection probability
    w       : simulation weight (nA)
    r       : rate (Hz)
    q is equal to tau_syn
    '''
    q = tau_syn
    var = (R**2*N*p*w**2*q**2*r)/(tau_m+tau_syn) * 0.001
    sigma = numpy.sqrt(var)
    return sigma

def ret_myfunc(x):
    '''
    computing e^(x^2) * (1+erf(x)) using loaded values
    load_myfunc() should be called earlier
    '''
    x = numpy.abs(x)
   
    epsilon = 0.0000000001
    
    if(x==0):
        return 0
    elif(x<100):
        initial = 0
        end = 100
        index = numpy.int((x+epsilon-initial)*10000)
        return myfunc_table[index]#[1]
    elif(x<200):
        initial = 100
        end = 200
        try:
            myfunc_table2.__len__()
        except NameError:        
            load_myfunc2()
            
        index = numpy.int((x+epsilon-initial)*10000)
        return myfunc_table2[index]#[1]
    else:
        raise NameError('Value of ' + str(x) + ' is out of range!')
    
    

def myerfc(x):
    '''
    computing 1+erf(x) using erfc(x)
    '''
    if x>=0:
        return 1+math.erf(x)
    else:
        return math.erfc(-x)
 
def myfunc(x):
    '''
    truncating function e^(x^2) * (1+erf(x))
    '''
    if x>-25:
        return numpy.exp(x**2)*myerfc(x)
    elif x<-200:
        return 0.0
    else:
        return ret_myfunc(x)    
    
def g(tau_ref,tau_m,R,V_th,V_reset,tau_syn,I,sigma,debug=False):
    '''
    Computing noisy gain function (g) for one point
    I        : Mean synaptic current (nA)
    sigma    : traditional std (=sqrt(2*sigma_v**2)), (mV)
    tau_ref in ms
    tau_m in ms
    tau_syn (q) in ms
    V_th in mV
    V_reset in mV
    R in Mohm
    '''    
    delta_V = V_th - V_reset
    alpha = 2.065   # for colored noise, set it to zero for white noise
    sqrt_pi = numpy.sqrt(numpy.pi)
    sqrt_taus = numpy.sqrt(1.0*tau_syn/tau_m)


    ub = (delta_V - R*I)/sigma + (alpha/2)*sqrt_taus
    lb = - R*I/sigma + (alpha/2)*sqrt_taus
        
    if debug:    
        print(lb, ub, sigma)
        
    temp, __ = scipy.integrate.quad(lambda x: myfunc(x), lb, ub)
        
    if(temp==numpy.inf):
        rate = 0
    else:
        rate = 1/(tau_ref + tau_m*sqrt_pi*temp)
            
    return 1000*rate    # return based on Hz
    
def noisyless_gain_function(tau_ref,tau_m,R,V_th,V_reset,tau_syn,I_max):
    '''
    Computing noisyless gain function with constant sigma
    tau_ref in ms
    tau_m in ms
    tau_syn (q) in ms
    V_th in mV
    V_reset in mV
    R in Mohm
    I in nA
    '''
    delta_V = V_th - V_reset
    I = numpy.arange(0,I_max,0.001) # nA
    rate = numpy.zeros(I.__len__())
    
    for i in range(I.__len__()):
        temp = R*I[i]/(R*I[i]-delta_V)
        if temp>0:
            rate[i] = 1/(tau_m*numpy.log(temp)+tau_ref)
        else:
            rate[i] = 0
  
    return I, 1000*rate
    

def modified_noisy_gain_function(tau_ref,tau_m,R,V_th,V_reset,tau_syn,sigma,I_max):
    '''
    Computing noisy gain function with constant sigma
    * stable version: combining with noisyless gain function in order to avoid approaching to max. firing rate  
    tau_ref in ms
    tau_m in ms
    tau_syn (q) in ms
    V_th in mV
    V_reset in mV
    R in Mohm
    I in nA
    '''
    I, rate_noisy, last = noisy_gain_function(tau_ref,tau_m,R,V_th,V_reset,tau_syn,sigma,I_max, return_last = True)

    if last!=0:
        I, rate_perfect = noisyless_gain_function(tau_ref,tau_m,R,V_th,V_reset,tau_syn,I_max)
        lag = rate_noisy[last-70] - rate_perfect[last-70]
        rate_noisy[last-70:] = rate_perfect[last-70:] + lag
    
    return I, rate_noisy

def noisy_gain_function(tau_ref,tau_m,R,V_th,V_reset,tau_syn,sigma,I_max, return_last = False):
    '''
    Computing noisy gain function with constant sigma
    * unstable version: may contain improper calculation and leads to approach to max. firing rate
    tau_ref in ms
    tau_m in ms
    tau_syn (q) in ms
    V_th in mV
    V_reset in mV
    R in Mohm
    sigma in mV
    I in nA
    '''
    
    last_correct = 0
    
    delta_V = V_th - V_reset
    alpha = 2.065   # for colored noise, set it to zero for white noise
    sqrt_pi = numpy.sqrt(numpy.pi)
    sqrt_taus = numpy.sqrt(1.0*tau_syn/tau_m)
    I = numpy.arange(0,I_max,0.001) # nA
    rate = numpy.zeros(I.__len__())
    #sigma = 0.5  # should be changed
    
    
    for i in range(I.__len__()):
        ub = (delta_V - R*I[i])/sigma + (alpha/2)*sqrt_taus
        lb = - R*I[i]/sigma + (alpha/2)*sqrt_taus
        
        if((ub<-200 or lb <-200) and last_correct==0):
            last_correct = i-1
            break
        
        #ub = numpy.max([-26,ub])
        #lb = numpy.max([-26,lb])
        #sigma += 0.01
        print(lb, ub, sigma)
        
        temp, __ = scipy.integrate.quad(lambda x: myfunc(x), lb, ub)
        
        if(temp==numpy.inf):
            rate[i]=0
        else:
            rate[i] = 1/(tau_ref + tau_m*sqrt_pi*temp)
        
    
    if return_last:
        return I, 1000*rate, last_correct
    else:
        return I, 1000*rate
        
    plt.figure()
    plt.plot(I,rate, linewidth=2)
    plt.grid()
    plt.show()



def noisy_gain_function_fortheta(tau_ref,tau_m,R,V_th,V_reset,tau_syn,sigma,I, return_last = False):
    '''
    Computing noisy gain function with flexibe sigma and I
    * unstable version: may contain improper calculation and leads to approach to max. firing rate
    tau_ref in ms
    tau_m in ms
    tau_syn (q) in ms
    V_th in mV
    V_reset in mV
    R in Mohm
    sigma in mV
    I in nA
    '''
    
    last_correct = 0
    
    delta_V = V_th - V_reset
    alpha = 2.065   # for colored noise, set it to zero for white noise
    sqrt_pi = numpy.sqrt(numpy.pi)
    sqrt_taus = numpy.sqrt(1.0*tau_syn/tau_m)
    #I = numpy.arange(0,I_max,0.001) # nA
    rate = numpy.zeros(I.__len__())
    #sigma = 0.5  # should be changed
    
    
    for i in range(I.__len__()):
        ub = (delta_V - R*I[i])/sigma[i] + (alpha/2)*sqrt_taus
        lb = - R*I[i]/sigma[i] + (alpha/2)*sqrt_taus
        
        if((ub<-106 or lb <-106) and last_correct==0):
            last_correct = i-1
            break
        
        #ub = numpy.max([-26,ub])
        #lb = numpy.max([-26,lb])
        #sigma += 0.01
        print(lb, ub, sigma[i])
        
        temp, __ = scipy.integrate.quad(lambda x: myfunc(x), lb, ub)
        
        if(temp==numpy.inf):
            rate[i]=0
        else:
            rate[i] = 1/(tau_ref + tau_m*sqrt_pi*temp)
        
    
    if return_last:
        return I, 1000*rate, last_correct
    else:
        return I, 1000*rate
        
    plt.figure()
    plt.plot(I,rate, linewidth=2)
    plt.grid()
    plt.show()


def count_depolarized_neuron(Voltage, size, spike_count, dep_threshold, duration, dt, ChR2_exp):
    '''
    count the number of neurons which fire or stay above 'dep_threshold' for at least 'duration' ms 
    '''
    
    DURATION = duration/dt
    counter_dep = 0
    counter_fire = 0
    spike_count_ave = 0
    
    for i in range(0,size):
        if ChR2_exp.__contains__(i):
            continue
        
        if spike_count[i]>0:
            counter_fire += 1
            spike_count_ave += spike_count[i]
        else:
            v_trace = Voltage.values[i]
            if numpy.nonzero(v_trace>dep_threshold)[0].__len__() > DURATION:
                counter_dep += 1
    
    return (spike_count_ave*1.0)/(size-ChR2_exp.__len__()),counter_fire, counter_dep
    
            

def plot_two_relation():
    R = 210 # Mohm
    tau_m = 37.6 # ms
    tau_syn = 7.4 # ms
    N = 90
    p = 0.5
    w = 0.05 # nA
    q = tau_syn  # ms
    r_h = 40 # Hz (1/s)
    I_max = 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    I = numpy.arange(0,1,0.01) # nA
    rate_2 = I/(N*p*q*w)  # rate based on network activity (kHz)
    
    sigma_2 = (2*R**2)/(2*(tau_m + tau_syn))*(N*p*q*w*r_h) * 0.001
    sigma = numpy.sqrt(sigma_2)
    
    print(sigma)
    
    __, rate = noisy_gain_function(4, tau_m, R, 24, 0, tau_syn, sigma, I_max) # rate based on noisy gain function (kHz)
    
    plt.plot(I,1000*rate_2, 'r')
    plt.plot(I,1000*rate)
    plt.show()
    

def rI_trace(N, rate, Isyn, bin_size):
    '''
    plot trace of (rate, synaptic current) for a population (not a single neuron)
    consider average of values over all neurons
    N        : Number of neurons
    rate     : rate_monitor object of Brian (Hz)
    I_syn    : Array of I_syn extracted by monitor object of Brian (nA)
    bin_size : bin size for rate (ms)
    '''
    
    Isyn_average = numpy.average(Isyn, axis=0)
    time = Isyn_average.__len__()
    bin_size = bin_size * 10  # multiplying by 1/dt
    
    I = numpy.zeros([time/bin_size])
    
    for i in range(0,time/bin_size):
        I[i] = numpy.average(Isyn_average[i*bin_size:(i+1)*bin_size])
        
    #plt.figure()
    return I, rate


def snapshot_voltage_distribution(voltages, size, Spike_counter, snapshot):
    '''
    Plot histogram of voltages of non-spikers in a special time 
    voltages      : Voltage object of Brian (made by StateMonitor)
    size          : size of population
    Spike_counter : Number of spike of each neuron (made by SpikeCounter of Brian)
    snapshot      : index of the snapshot
    '''
    counter = 0
    #sum_v = numpy.zeros(voltages.values[0].__len__())
    for i in range(0,size):
        if Spike_counter[i]==0:
            counter += 1
            #sum_v += voltages.values[i]
   
    print("firing:" +str(size - counter))
    
    if counter==0:
        return
    
    volt = numpy.zeros(counter)
    k=0
    for i in range(0,size):
        if Spike_counter[i]==0:
            #print k
            volt[k] = voltages.values[i][numpy.int(snapshot)] * 1000  # 1000 : since based on mV
            k += 1
    
    plt.figure()
    plt.hist(volt,bins=30)
    plt.title("Nonhubs: Distribution of non-firing neurons voltage in t=" + str(snapshot/10))    

def mean_voltage_non_exp(voltages, size, myindex, chr2_index):
    '''
    Calculate mean voltage of non-expressing neurons
    voltages      : Voltage object of Brian (made by StateMonitor)
    size          : size of population
    myindex       : indices of neurons of desired subpopulations
    chr2_index    : indices of expressing neurons
    '''
    
    counter = 0
    sum_v = numpy.zeros(voltages.values[0].__len__())

    for i in range(0,size):
        if myindex.__contains__(i) and chr2_index.__contains__(i) == False:
            sum_v += voltages.values[i]
            counter += 1
            
    return sum_v/counter
        
def mean_voltage(voltages, size, Remove_spiker=False, Spike_counter=None):
    '''
    Calculating mean voltage of a population
    voltages      : Voltage object of Brian (made by StateMonitor)
    size          : size of population
    Remove_spiker : Remove spiker neurons from averaging
    Spike_counter : Number of spike of each neuron (made by SpikeCounter of Brian)
    '''
    if Remove_spiker:
        counter = 0
        sum_v = numpy.zeros(voltages.values[0].__len__())
        for i in range(0,size):
            if Spike_counter[i]==0:
                counter += 1
                sum_v += voltages.values[i]
        print(counter)
        return sum_v/counter
    else:
        return numpy.mean(voltages.values, axis=0)

def two_relations_trajectory(I_syn, rate, default_dt, new_dt):
    '''
        finding trajectory of (I_syn,rate) for a single neuron
        I_syn     : synaptic input (sum of Exc and Inh current), discretized using default_dt
        rate      : rate trajectory of the neuron, discretized using new_dt
        defult_dt : time step used by simulator
        new_dt    : time step for making the trajectory
    '''
    
    k = new_dt/default_dt
    
    l = len(I_syn)
    
    if l%k != 0:
        I_syn = numpy.append(I_syn, numpy.zeros(l%k))
        
    temp = numpy.reshape(I_syn, (l/k,k))
    dis_I_syn = numpy.average(temp, 1)

    fig, ax = plt.subplots()
    
    
    ax.plot(dis_I_syn, rate, '-o')
    
    plt.show()
    
def replace_AP(Voltage, Spike_times, AP_shape):
    len = AP_shape.__len__()
    for i in Spike_times:
        Voltage[i*10000:i*10000+len] = AP_shape
        
    return Voltage
    
def updown_duration(rate, threshold, zero_one = False, window_len = 400, sigma = 120):
    '''
    returns durations of up and down states, each one in a array
    Uses population rate (particularly rate of Exc (hubs))
    '''
    
    filter_x = numpy.arange(-window_len,window_len,1)
    filter = 1/(numpy.sqrt(2*numpy.pi)*sigma)*numpy.exp(-filter_x**2/(2*sigma**2))  # Gaussian filter
    filter = filter/numpy.sum(filter)  # normalizing filter
    smooth_rate = numpy.convolve(rate, filter, mode='same')
    
    #plt.figure()
    #plt.plot(rate)
    #plt.plot(smooth_rate, lw=2)
    
    dif = numpy.diff(1.0*(smooth_rate>threshold))
    #plt.stem(range(0,dif.__len__()),dif)
    go_up_time = numpy.where(dif==1)[0]
    go_down_time = numpy.where(dif==-1)[0]
    
    down_duration = numpy.zeros(go_up_time.__len__()-1)
    up_duration = numpy.zeros(go_down_time.__len__())
    
    for i in range(up_duration.__len__()):
        up_duration[i] = go_down_time[i] - go_up_time[i]
    
    for i in range(down_duration.__len__()):
        down_duration[i] = go_up_time[i+1] - go_down_time[i]
    
    
    #plt.show()
     
    if zero_one:
        return up_duration, down_duration, smooth_rate>threshold
    else:
        return up_duration, down_duration 
    
    
def updown_duration2(rate, threshold, zero_one = False, window_len = 400, sigma = 120):
    '''
    returns durations of up and down states, each one in a array
    Uses population rate (particularly rate of Exc (hubs))
    '''
    
    filter_x = numpy.arange(-window_len,window_len,1)
    filter = 1/(numpy.sqrt(2*numpy.pi)*sigma)*numpy.exp(-filter_x**2/(2*sigma**2))  # Gaussian filter
    filter = filter/numpy.sum(filter)  # normalizing filter
    smooth_rate = numpy.convolve(rate, filter, mode='same')
    
    dif = numpy.diff(1.0*(smooth_rate>threshold))
    #plt.stem(range(0,dif.__len__()),dif)
    #plt.show()
    go_up_time = numpy.where(dif==1)[0]
    go_down_time = numpy.where(dif==-1)[0]
    
    down_duration = numpy.zeros(go_up_time.__len__()-1)
    up_duration = numpy.zeros(go_down_time.__len__())
    
    for i in range(up_duration.__len__()):
        up_duration[i] = go_down_time[i] - go_up_time[i]
    
    for i in range(down_duration.__len__()):
        down_duration[i] = go_up_time[i+1] - go_down_time[i]
    
    
    if zero_one:
        return up_duration, down_duration, smooth_rate>threshold, smooth_rate
    else:
        return up_duration, down_duration  


def updown_switch_times(voltage, threshold, window_len = 400, sigma = 70):
    '''
    returns the time of switching states (from up to down and inverse) based on the voltage
    '''
    
    filter_x = numpy.arange(-window_len,window_len,1)
    filter = 1/(numpy.sqrt(2*numpy.pi)*sigma)*numpy.exp(-filter_x**2/(2*sigma**2))  # Gaussian filter
    filter = filter/numpy.sum(filter)  # normalizing filter
    smooth_rate = numpy.convolve(voltage, filter, mode='same')
    
    dif = numpy.diff(1.0*(smooth_rate>threshold))

    go_up_time = numpy.where(dif==1)[0]
    
    #plt.figure()
    #plt.title('inside analysis')
    #plt.plot(voltage[:100000], lw=2, color='blue')
    #plt.plot(smooth_rate[:100000], lw=2, color='green')
    #plt.show()
    
    go_down_time = numpy.where(dif==-1)[0]
    
    return go_up_time, go_down_time


def updown_switch_times_tmp(voltage, threshold, window_len = 400, sigma = 70):
    '''
    returns the time of switching states (from up to down and inverse) based on the voltage
    '''
    
    filter_x = numpy.arange(-window_len,window_len,1)
    filter = 1/(numpy.sqrt(2*numpy.pi)*sigma)*numpy.exp(-filter_x**2/(2*sigma**2))  # Gaussian filter
    filter = filter/numpy.sum(filter)  # normalizing filter
    smooth_rate = numpy.convolve(voltage, filter, mode='same')
    
    dif = numpy.diff(1.0*(smooth_rate>threshold))

    go_up_time = numpy.where(dif==1)[0]
    
    #plt.figure()
    #plt.title('inside analysis')
    #plt.plot(voltage[:100000], lw=2, color='blue')
    #plt.plot(smooth_rate[:100000], lw=2, color='green')
    #plt.show()
    
    go_down_time = numpy.where(dif==-1)[0]
    
    return go_up_time, go_down_time, smooth_rate
    
def weighted_array_to_single_array(array, weights):
    '''
    Converts a weighted array to the array repetitive elements based on weights
    '''    
    
    output = numpy.zeros(numpy.sum(weights))
    k = 0
    
    for i in range(0,weights.__len__()):
        for j in range(0,weights[i]):
            output[k] = array[i]
            k += 1
            
    return output    
    
    
def plot_histogram(values,bins, xlabel='' , ylabel='' , ax = None):
    # get the corners of the rectangles for the histogram
    left = numpy.array(bins[:-1])
    right = numpy.array(bins[1:])
    bottom = numpy.zeros(len(left))
    top = bottom + values


    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = numpy.array([[left,left,right,right], [bottom,top,top,bottom]]).T

    # get the Path object
    barpath = matplotlib.path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = matplotlib.patches.PathPatch(barpath, facecolor='blue', edgecolor='black', alpha=0.8)
    
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    #ax.set_xticks([0,1000,2000,3000,4000,5000])
    #ax.set_xticklabels(['0','100','200','300','400','500'])
    #ax.set_xlabel('time(ms)')
    #ax.set_ylabel('Firing rate(Hz)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    

def smooth_function(func, window_len):
    filter = numpy.ones(window_len)
    filter = filter/numpy.sum(filter)  # normalizing filter
    smooth_func = numpy.convolve(func, filter, mode='same')
    return smooth_func



def CC(a,b,std_a,std_b,len):
    '''
    computing cross-correlation between two signals a and b
    '''
    
    tmp = numpy.sum(numpy.multiply(a,b))
    return tmp/(len*std_a*std_b)
    

def toSpTr(spiketimes, len):
    '''
    converting spiketimes to spike train with 10ms bin
    len :    duration of simulation (ms)
    '''
    sptrain = numpy.zeros(len/10) # bin = 10ms
    for i in spiketimes:
        sptrain[int(i*1000.0/10)] += 1 # bin = 10ms
    
    return sptrain


def allCC(exc_spiketimes, inh_spiketimes, start_time, end_time, len):
    '''
    computing cross-correlation between all pairs
    exc_spiketimes :    spiketimes of excitatory neurons
    inh_spiketimes :    spiketimes of inhibitory neurons
    correlations will be computed between start_time (ms) and end_time (ms)
    len:                duration of simulation (ms)
    '''
    
    limited_len = (end_time - start_time)/10
    size_exc = exc_spiketimes.__len__()
    size_inh = inh_spiketimes.__len__()
    

    std_exc = numpy.zeros(size_exc)
    std_inh = numpy.zeros(size_inh)

    spike_exc = []
    spike_inh = []
 

    for i in range(0, size_exc):
        spike_exc.append(toSpTr(exc_spiketimes[i], len)[start_time/10: end_time/10])
        std_exc[i] = numpy.std(spike_exc[i])
        spike_exc[i] -= numpy.mean(spike_exc[i])

    
    for i in range(0, size_inh):
        spike_inh.append(toSpTr(inh_spiketimes[i], len)[start_time/10: end_time/10])
        std_inh[i] = numpy.std(spike_inh[i])
        spike_inh[i] -= numpy.mean(spike_inh[i])
     
    exc_cc = []
    inh_cc = []
    all_cc = []
     
    for i in range(size_exc):
        for j in range(i+1, size_exc):
            tmp = CC(spike_exc[i], spike_exc[j], std_exc[i], std_exc[j], limited_len)
            if not numpy.isnan(tmp):
                exc_cc.append(tmp)        
                all_cc.append(tmp)
                
    
    for i in range(size_inh):
        for j in range(i+1, size_inh):
            tmp = CC(spike_inh[i], spike_inh[j], std_inh[i], std_inh[j], limited_len)
            if not numpy.isnan(tmp):
                inh_cc.append(tmp)        
                all_cc.append(tmp)
                
    
    for i in range(size_exc):
        for j in range(size_inh):
            tmp = CC(spike_exc[i], spike_inh[j], std_exc[i], std_inh[j], limited_len)
            if not numpy.isnan(tmp):
                all_cc.append(tmp)
                
    
    return exc_cc, inh_cc, all_cc

'''
generating first current using ou-process
'''

def gen_ou(plot=False):
    I0 = 10.0  # nA
    tau = 10.0  # ms
    Delta_T = 1.0/10  #ms, sampling freq = 20 kHz
    sigma_0 = 1.0  # unitless
    Delta_sigma = 2.0  # unitless
    f = 0.2 * 0.001  # kHz 
    
    
    len_current = 10000  # ms
    len = int(len_current / Delta_T)  # length of arrays: I and time
    I = numpy.zeros(len)  # nA
    time = numpy.arange(0, len_current, Delta_T)  # ms
    sigma = numpy.zeros(len)  # unitless
    
    I[0] = 0.0
    for counter in range(1, len):
        #print 2*numpy.pi*f*time[counter], numpy.sin(2*numpy.pi*f*time[counter])
        sigma[counter] = sigma_0 * (1 + Delta_sigma * numpy.sin(2*numpy.pi*f*time[counter]))
        I[counter] = I[counter-1] + (I0 - I[counter-1])/tau*Delta_T + numpy.sqrt(2*sigma[counter]**2*Delta_T/tau)*numpy.random.normal()  # N(0,1)
    
    if plot:
        plt.plot(time, I, lw=2)
    
        plt.plot(time, I0*(1-numpy.exp(-time/tau)), color='red', lw=2)
    
        plt.figure()
        plt.plot(time, sigma, color='green', lw=2)
        plt.grid()
    
        plt.show()

    return time, I