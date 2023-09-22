# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:36:19 2023

Created for the EITN 2023 Fall School, NEURON project.
Code describing the baseline state of a L2/3 V1 pyramidal neuron,
featuring both background (noise) and stimulus-driven synaptic input.

Test version: Synapses and stim-driven stimulation have been removed, 
to test whether the model functions in its simplest state.

@author: KEPetousakis
"""

#%%===================================== Import Statements =====================================

from __future__ import division
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import datetime
import utility_spikes_pickling as util

#%%===================================== Initialization =====================================

# This code handles import of some required functions.


# This code loads the cell morphology and imports active and synaptic mechanisms.
# It automatically checks for your OS and attempts to load the appropriate files,
# given that you have compiled the mechanism files properly (see README for details).
# If this code fails to run properly, you will get an error stating "na is not a MECHANISM".
h.load_file("stdgui.hoc")  # Load required base NEURON libraries
h.load_file("hoc.files/neuron1_modified.hoc")  # Load morphology
if os.name == 'nt':
	h.nrn_load_dll("mod.files/nrnmech.dll")  # this needs to be added on Windows, or custom mechanisms won't work
else:
	h.nrn_load_dll("mod.files/x86_64/libnrnmech.so")  # this needs to be added on Linux/Unix/MacOs, or custom mechanisms won't work

try:
	if os.name == 'posix':
		h.nrn_load_dll("mod.files/arm64/special.nrn")  # This needs to be added on MacOS (if NEURON works at all on your Mac...), or custom mechanisms won't work.
except:
	pass

h.load_file("hoc.files/cell_setup.hoc")  # Set up active and passive (non-synaptic) membrane mechanisms, and set currents to equilibrium (calls 'current-balance.hoc')	

# Setting up easy-to-use handles for neuronal Sections
soma = h.soma
apical = h.apic
basal = h.dend


#%%=========================== Functionality testing ================================

# You may use this code, if you want to, to verify that the model functions properly.
# It adds a current clamp in the middle of the soma that pulses twice, and 
# should normally cause the soma to produce a single action potential after the second 
# pulse from the clamp.


# Current injection
stim1 = h.IClamp(soma(0.5))  # add a current clamp the the middle of the soma
stim1.delay = 500  # ms
stim1.dur = 10 # ms
stim1.amp = 0.33 # nA

stim2 = h.IClamp(soma(0.5))  # add a current clamp the the middle of the soma
stim2.delay = 520  # ms
stim2.dur = 10 # ms
stim2.amp = 0.33 # nA

# set up recordings
soma_v = h.Vector()  # set up a recording vector
soma_v.record(soma(0.5)._ref_v)  # record voltage at the middle of the soma
t = h.Vector()
t.record(h._ref_t)  # record time

# run simulation
h.v_init = -79  # set starting voltage
h.dt = 0.1
h.steps_per_ms = 10
h.tstop = 1000  # set simulation time 
h.run();  # run simulation

fig = plt.figure(figsize=(8, 5))
plt.plot(t, soma_v, color='k', label='soma(0.5)')
plt.xlim(0, 1000)
plt.xlabel('Time (ms)', fontsize=15)
plt.ylabel('Voltage (mV)', fontsize=15)
plt.legend(fontsize=14, frameon=False)
fig.set_tight_layout(True)
figmanager = plt.get_current_fig_manager()
figmanager.window.showMaximized()
plt.show()

input()  # Wait for input here (simply to control program flow)

