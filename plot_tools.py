# This code has been modified from that available at 
# the exercises for the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

# Started by Sharbat, 11 Jun 2018

import matplotlib.pyplot as plt
import brian2 as b2
import numpy


def plot_voltage_and_current_traces(voltage_monitor, spike_monitor, current, title=None, legend_location=20):
    """plots voltage and current .
    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current (TimedArray): injected current
        title (string, optional): title of the figure
        firing_threshold (Quantity, optional): if set to a value, the firing threshold is plotted.
        legend_location (int): legend location. default = 0 (="best")
    Returns:
        the figure
    """

    assert isinstance(voltage_monitor, b2.StateMonitor), "voltage_monitor is not of type StateMonitor"
    assert isinstance(current, b2.TimedArray), "current is not of type TimedArray"

    time_values_ms = voltage_monitor.t / b2.ms

    # current
    axis_c = plt.subplot(211)
    c = current(voltage_monitor.t, 0)
    max_current = max(current(voltage_monitor.t, 0))
    min_current = min(current(voltage_monitor.t, 0))
    margin = 1.05 * (max_current - min_current)
    # plot the input current time-aligned with the voltage monitor
    plt.plot(time_values_ms, c, "r", lw=2)
    if margin > 0.:
        plt.ylim((min_current - margin) / b2.amp, (max_current + margin) / b2.amp)
    # plt.xlabel("t [ms]")
    plt.ylabel("Input current [A] \n min: {0} \nmax: {1}".format(min_current, max_current))
    plt.grid()
    axis_v = plt.subplot(212)

    volts = voltage_monitor[0].v

    for i in spike_monitor.t:
        index = int(i/b2.ms*10)
        volts[index] = 10*b2.mV

    plt.plot(time_values_ms, volts / b2.mV, lw=2)
    # plt.plot(time_values_ms, voltage_monitor[0].vt / b2.mV, lw=2)
    # for t in spike_monitor.t:
    #     axis_v.axvline(t/b2.ms, ls='--', c='C1', lw=1)
    max_vval = max(volts)
    max_vtaval = max(voltage_monitor[0].vt)
    min_vval = min(volts)
    min_vtval = min(voltage_monitor[0].vt)
    max_val = max(max_vval,max_vtaval)
    min_val = min(min_vval,min_vtval)
    margin = 0.05 * (max_val - min_val)
    plt.ylim((min_val - margin) / b2.mV, (max_val + margin) / b2.mV)
    plt.xlabel("t [ms]")
    plt.ylabel("membrane voltage [mV]\n min: {0}\n max: {1}".format(min_val, max_val))
    plt.grid()

    # plt.legend(["v", "vt"], fontsize=12, loc=legend_location)

    if title is not None:
        plt.suptitle(title)