# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:36:19 2023

Created for the EITN 2023 Fall School, NEURON project.
Code describing the baseline state of a L2/3 V1 pyramidal neuron,
featuring both background (noise) and stimulus-driven synaptic input.

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
# Pseudoglobal variables
_current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_current_date = datetime.datetime.now().strftime("%Y-%m-%d")
_LOGGING = True  # Moves (most) terinal/commandline outputs to a log file
_VERBOSE = False # Disables (most) terminal/commandline print() statements
_pickle_save = False # Disables saving output data (somatic voltage trace) as a "pickle" file (saves as .dat instead)
_test_function = False # Toggles functionality testing (with a current clamp at the soma) to ensure the core of the model functions properly.

# Default launch arguments (used through the terminal/command line)
# 'nrn': Neuron ID - changing it changes the distribution of synapses onto the neuron (0 to 9, in steps of 1, other positive integers also supported)
# 'run': Run (simulation) ID - changing it changes the synaptic spike trains (0 to 9, in steps of 1, other positive integers also supported)
# 'stim': Orientation of presented stimulus (0 to 90 in steps of 10, degrees)
# 'att': Toggle attentional inputs (converts a fraction of apical inputs into differentially-modulated "attentional" inputs)
# 'batch': Toggle execution mode ("single run mode":0, "multiple runs mode":1 - to be used with "concurrent_runs.py")
_ARGLIST = ['nrn', 'run', 'stim', 'att', 'batch']
_ARGVALS = {'nrn':0, 'run':0, 'stim':0, 'att':0, 'batch':0}

print("Launch arguments: ", end=''); print(*sys.argv)  # Reminder: star symbol (*) unpacks a list (here, "sys.argv")

# Parse launch arguments (if any are provided, otherwise fall back to default values)
try:
	if len(sys.argv) <= 1:
		raise Exception		
	else:
		for i, arg in enumerate(sys.argv):
			if i > 0:
				for valid_arg in _ARGLIST:
					if valid_arg in arg:
						_ARGVALS[valid_arg] = int(arg[len(valid_arg)+1:])
except Exception as e:
	_ARGVALS = {'nrn':0, 'run':0, 'stim':0, 'att':0, 'batch':0}

# Setting up a simulation identifier to be used to name output files differently depending on the launch arguments
simulation_id = f'n{_ARGVALS["nrn"]:.0f}r{_ARGVALS["run"]:.0f}s{_ARGVALS["stim"]:.0f}a{_ARGVALS["att"]:.0f}'

_multiple_runs = True if _ARGVALS['batch'] else False

def log(text_str):
	"""Logging function, to record details from the simulation setup and execution"""
	if _LOGGING:
		with open(f'logs/log_{_current_date}_{simulation_id}.dat', 'a+') as f:
			f.write(f'[{datetime.datetime.now()}] | {text_str}\n')
		return
	else:
		return
		
	
def mprint(arg):
	"""Modular Print (mprint) function, that allows suppressing all print statements by changing the global
	variable _VERBOSE"""
	if _VERBOSE:
		print(arg)
		return
	else:
		if _LOGGING:
			log(arg)
		else:
			return

# This code handles import of some required functions.
# If only running the simulation once (_multiple_runs = False),
# then the normal GUI will be loaded, and the NEURON main menu will be visible.
# Otherwise, NEURON will run silently in the background without GUI elements.
if not _multiple_runs:
	from neuron import gui
else:
	h.load_file("stdgui.hoc")


# This code loads the cell morphology and imports active and synaptic mechanisms.
# It automatically checks for your OS and attempts to load the appropriate files,
# given that you have compiled the mechanism files properly (see README for details).
# If this code fails to run properly, you will get an error stating "na is not a MECHANISM".
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

if not _multiple_runs:
	h.load_file("hoc.files/basic-graphics.hoc")  # Loads custom  NEURON library for real-time voltage trace display

# Control variables
bg = 1                  # Toggle background (noise) on-off
bst = 1					# Toggle basal stim-driven inputs on-off
ast = 1					# Toggle apical stim-driven inputs on-off
att = int(_ARGVALS['att'])					# Toggle attentional inputs on-off (a fraction of apical inputs is converted)
afbg = 0                # Percentage of apical background synapses "converted" to attentional inputs
afst = 0                # Percentage of apical stimulus-driven synapses "converted" to attentional inputs
afreq = 0.2             # Stimulation frequency of attentional inputs (for the Poisson spike train)
stim_factor = 1.2         # Multiplicative scaling factor for stim-driven synapse activation frequency (minimizes need for multiple runs if >1)

stimulus_base_frequency   = 0.3     # Frequency of stimulus-driven input events (parameter of a Poisson event generator)
noise_base_frequency      = 0.11    # Frequency of background-driven input events (parameter of a Poisson event generator)
inhibition_base_frequency = 0.11    # Frequency of (background) inhibitory input occurrence (parameter of a Poisson event generator)

stim_orient = int(_ARGVALS['stim'])  # Assign the orientation of the stimulus
nrn = int(_ARGVALS['nrn'])           # Assign neuron ID (see Random Number Generator section for details on why we need this)
run = int(_ARGVALS['run'])           # Assign simulation ID (see Random Number Generator section for details on why we need this)

# Setting up easy-to-use handles for neuronal Sections
soma = h.soma
apical = h.apic
basal = h.dend

# Required simulation variables
sim_time = 2500         # tstop for the simulation (total simulation time, in ms)
delta = 0.1				# dt for the simulation (simulation/integration time step, in ms)
n_neuron = nrn
n_run = run
h.steps_per_ms = int(1/delta)  # NEURON needs this to be assigned, otherwise the 'h.dt' parameter below may revert to the default value
h.dt = delta
h.tstop = sim_time  # Set the total simulation runtime (ms)


#%%=========================== Functionality testing ================================

# You may use this code, if you want to, to verify that the model functions properly.
# It adds a current clamp in the middle of the soma that pulses twice, and 
# should normally cause the soma to produce a single action potential after the second 
# pulse from the clamp.

if _test_function:

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
	raise Exception # You should not continue on to a regular simulation after running a function test - set '_test_function' to 'False' first!
	                # The Exception is an (inelegant) way to stop execution past this point.

#%%===================================== Model Setup =====================================
# The following part achieves the following goals:
#	-Calculating numbers of synapses (of different types, background vs stim-driven,
#	 as well as excitatory vs inhibitory)
#	-Distributing all synapses onto the dendrites using a length-based approach
# 	-Allocating the synapses onto the actual dendrites
#	-Linking synapses to spike trains, so that they are activated rather than quiescent
# This is the "heart" of the model, and thus the most complicated part.
# For each of the above goals, the first instance of code will be documented.

#%%=========================== Calculating synapses ================================
# The numbers (percentages) in this section are taken from literature.
# For details, see (Park, Papoutsi et al., 2019)

tl = 3298.1508 						    #Total Length of Neuron
syn_exc_total = int(tl*2)				#Assume synaptic density 2/micrometer     
mprint(f"Total number of synapses: {syn_exc_total}")
syn_exc = int(0.75*syn_exc_total)		#25% of spines are visual responsive, Chen et al. 2013, rest set as background input.
syn_basal = int(0.6*syn_exc) 	        #deFelipe, Farinas,1998
syn_apical = int(syn_exc - syn_basal)   #deFelipe, Farinas,1998

mprint(f"(Background)\t Basal excitatory: {syn_basal}\t | Apical excitatory: {syn_apical}")

syn_inh = int(syn_exc_total*0.15) 			    #15% Binzegger, Martin, 2004 (L2/3, cat V1)  
syn_inh_soma = int(syn_inh*0.07)		        #deFelipe, Farinas,1998
syn_inh_apical = int(syn_inh*0.33)   
syn_inh_basal = syn_inh - syn_inh_soma - syn_inh_apical

mprint(f"(Background)\t Basal inhibitory: {syn_inh_basal}\t | Apical inhibitory: {syn_inh_apical}\t | Somatic inhibitory: {syn_inh_soma}")

total_syn = int(0.25*syn_exc_total)			    #25% of spines are visual responsive, Chen et al. 2013
total_syn_basal = int(total_syn*0.6)            #60% of synapses reach the basal tree (deFelipe, Farinas, 1998)
total_syn_apic = total_syn - total_syn_basal

mprint(f"(Stimulus)\t\t Basal excitatory: {total_syn_basal}\t | Apical excitatory: {total_syn_apic}")


#%%=========================== Allocating synapses ================================
# Tuning is achieved via a Gaussian distribution of synaptic orientation preferences onto the dendrites.
# A Gaussian distribution (known also as a "Bell Curve") features the following parameters:
# x: the independent variable, in this case it is the synaptic orientation preference
# μ (mu): the mean of the distribution, or the synaptic orientation preference with the highest probability/frequency
# σ (sigma): the standard deviation of the distribution, or the width of the distribution of synaptic orientation preferences

# Set tuning-related parameters
apical_width = 30  # Width of the distribution of synaptic orientation preferences for the apical tree
basal_width = 30   # Width of the distribution of synaptic orientation preferences for the basal tree
tag_basal = 0      # Mean of the distribution of synaptic orientation preferences for the basal tree
tag_apical = 0     # Mean of the distribution of synaptic orientation preferences for the apical tree

# Set synaptic (AMPA/NMDA/GABAa) conductances
ampa_g = 0.00084
nmda_g = 0.00115
gaba_g = 0.00125

# Set random number seeds
# These need to be carefully set up, otherwise results are non-reproducible!
rng_seed1 = basal_width*apical_width*(tag_basal+1)*10000*(n_run+1)*(n_neuron+1)
rng_seed2 = basal_width*apical_width*(tag_basal+1)*10000*(n_neuron+1)	

# (Approximately) Distribute background synapse numbers according to section length
apical_L = sum([x.L for x in apical])
basal_L = sum([x.L for x in basal])
apical_nsyn_bg_inh = [int(syn_inh_apical*x.L/apical_L) for x in apical]
basal_nsyn_bg_inh = [int(syn_inh_basal*x.L/basal_L) for x in basal]
apical_nsyn_bg_exc = [int(syn_apical*x.L/apical_L) for x in apical]
basal_nsyn_bg_exc = [int(syn_basal*x.L/basal_L) for x in basal]

# (Approximately) Distribute stimulus synapse numbers according to section length
apical_nsyn_stim = [int(total_syn_apic*x.L/apical_L) for x in apical]
basal_nsyn_stim = [int(total_syn_basal*x.L/basal_L) for x in basal]

# Background excitation/inhibition parameters
freq_exc = noise_base_frequency
freq_inh = inhibition_base_frequency
offset_seed = 1200

# Random number generators for position and event timings
# Again, these need to be set up carefully to ensure reproducibility!	
rng_pos = h.Random(rng_seed2)
rng_pos.uniform(0,1)
rng_t_exc = h.Random(rng_seed1+offset_seed)
rng_t_exc.poisson(freq_exc/1000)
rng_t_inh = h.Random(rng_seed1-offset_seed)
rng_t_inh.poisson(freq_inh/1000)

# Toggle background components on-off
# Here, you could toggle specific background-driven synapse groups on/off.
# 'bg': background (independent of stimulus orientation)
# 'exc': excitatory (AMPA and NMDA)
# 'inh': inhibitory (GABAa)
background_enabled = bg
if background_enabled:
	basal_bg_exc_enabled = True
	basal_bg_inh_enabled = True
	apical_bg_exc_enabled = True
	apical_bg_inh_enabled = True
	soma_bg_inh_enabled = True
else:
	basal_bg_exc_enabled = False
	basal_bg_inh_enabled = False
	apical_bg_exc_enabled = False
	apical_bg_inh_enabled = False
	soma_bg_inh_enabled = False

#%%================================ Background synapses ================================

bg_fraction_attention_driven = afbg           # Fraction of background-driven synapses to be allocated to the "attentional" pool
bg_number_attention_driven_persection = []    # List to hold the number of synpases allocated to the "attentional" pool, per dendrite
bg_number_noise_driven_persection = []        # List to hold the remaining, background-driven synapses

# Basal excitatory background
# Comments here are applicable to ALL background synaptic allocation code
if basal_bg_exc_enabled:           # Initialize lists that hold spike trains and synapses
	events_bg_exc_basal = []       # Presynaptic events
	vecstims_bg_exc_basal = []     # Synaptic activations (spike trains)
	syn_bg_ampa_basal = []         # AMPA synapse objects
	syn_bg_nmda_basal = []         # NMDA synapse objects
	conn_bg_ampa_basal = []        # Connection objects for AMPA synapses
	conn_bg_nmda_basal = []        # Connection objects for NMDA synapses
	                               # Reminder: Inhibitory synapses are only GABAa (found in subsequent code)
	
	for i,sec in enumerate(basal):  # Iterate through basal dendritic sections
		sec  # Access/select the section that the "for" loop has reached
		for isyn in range(0, basal_nsyn_bg_exc[i]):  # Iterate through all synapses to be allocated to this particular dendritic section
			events_bg_exc_basal.append(h.Vector())   # Create a vector (list of numbers) that will contain the spike train
			vecstims_bg_exc_basal.append(h.VecStim())  # Create a VecStim object to "play back" the spike train and activate the synapses
			for time in range(0, int(h.tstop)):      # Iterate through the duration of the simulation as discrete time points
				if rng_t_exc.repick() > 0:           # If the poisson process defined in "rng_t_exc" returns a value greater than 0 (0 = no event, 1 or more= that many events)...
					events_bg_exc_basal[-1].append(time)  # ...append the current time onto the spike train, as it contains an "event"
			vecstims_bg_exc_basal[-1].delay = 0  # The spike train does not have an initial delay
			vecstims_bg_exc_basal[-1].play(events_bg_exc_basal[-1]) # "Play back" the spike trains we created through the VecStim object
			syn_pos = rng_pos.repick()   # Choose a new random position for the synapse we are creating (0 ... 1, floating-point number, 0=origin of section, 1=end of section)
			syn_bg_ampa_basal.append(h.GLU(sec(syn_pos)))  # Append a new AMPA synapse to the list of synapses
			syn_bg_nmda_basal.append(h.nmda(sec(syn_pos))) # Append a new NMDA synapse to the list of synapses
			conn_bg_ampa_basal.append(h.NetCon(vecstims_bg_exc_basal[-1], syn_bg_ampa_basal[-1], -20, 0, ampa_g)) # Link the VecStim to the AMPA synapse, and set properties
			conn_bg_nmda_basal.append(h.NetCon(vecstims_bg_exc_basal[-1], syn_bg_nmda_basal[-1], -20, 0, nmda_g)) # Link the VecStim to the NMDA synapse, and set properties
	mprint("Allocated basal background excitatory synapses.")
			
# Basal inhibitory background
# Above comments apply here as well, with the exception that we have GABAa synapses, not AMPA and NMDA.
if basal_bg_inh_enabled:
	events_bg_inh_basal = []
	vecstims_bg_inh_basal = []
	syn_bg_gaba_basal = []
	conn_bg_gaba_basal = []
	
	for i,sec in enumerate(basal):
		sec
		for isyn in range(0, basal_nsyn_bg_inh[i]):
			events_bg_inh_basal.append(h.Vector())
			vecstims_bg_inh_basal.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_inh.repick() > 0:
					events_bg_inh_basal[-1].append(time)
			vecstims_bg_inh_basal[-1].delay = 0
			vecstims_bg_inh_basal[-1].play(events_bg_inh_basal[-1])
			syn_pos = rng_pos.repick()
			syn_bg_gaba_basal.append(h.GABAa(sec(syn_pos)))
			conn_bg_gaba_basal.append(h.NetCon(vecstims_bg_inh_basal[-1], syn_bg_gaba_basal[-1], -20, 0, gaba_g))
	mprint("Allocated basal background inhibitory synapses.")


# Somatic inhibitory background (add into the "background synapses" cell)
# Well spotted! This was indeed missing. Does it fix the issue though...?
if soma_bg_inh_enabled:
	events_bg_inh_soma = []
	vecstims_bg_inh_soma = []
	syn_bg_gaba_soma = []
	conn_bg_gaba_soma = []
	
	# We do not need to iterate over sections here, since the soma is one section (made up of many segments, but it is still one section)
	# This means we don't need a "for" loop to iterate through sections, but only a "for" loop that iterates over the synapses that we are
	# allocating.
	soma
	for isyn in range(0, syn_inh_soma):
		events_bg_inh_soma.append(h.Vector())
		vecstims_bg_inh_soma.append(h.VecStim())
		for time in range(0, int(h.tstop)):
			if rng_t_inh.repick() > 0:
				events_bg_inh_soma[-1].append(time)
		vecstims_bg_inh_soma[-1].delay = 0
		vecstims_bg_inh_soma[-1].play(events_bg_inh_soma[-1])
		syn_pos = rng_pos.repick()
		syn_bg_gaba_soma.append(h.GABAa(sec(syn_pos)))
		conn_bg_gaba_soma.append(h.NetCon(vecstims_bg_inh_soma[-1], syn_bg_gaba_soma[-1], -20, 0, gaba_g))
	mprint("Allocated somatic background inhibitory synapses.")

		
# Apical excitatory background
if apical_bg_exc_enabled:
	events_bg_exc_apical = []
	vecstims_bg_exc_apical = []
	syn_bg_ampa_apical = []
	syn_bg_nmda_apical = []
	conn_bg_ampa_apical = []
	conn_bg_nmda_apical = []
	
	for i,sec in enumerate(apical):
		sec
		bg_number_attention_driven_persection.append(int(apical_nsyn_bg_exc[i]*bg_fraction_attention_driven))
		bg_number_noise_driven_persection.append(apical_nsyn_bg_exc[i] - bg_number_attention_driven_persection[-1])
		mprint(f'\tApical {i} background excitatory synapses ({apical_nsyn_bg_exc[i]}) split: {bg_number_attention_driven_persection[-1]} attention-driven, {bg_number_noise_driven_persection[-1]} background.')
		for isyn in range(0, bg_number_noise_driven_persection[-1]):
			events_bg_exc_apical.append(h.Vector())
			vecstims_bg_exc_apical.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_exc.repick() > 0:
					events_bg_exc_apical[-1].append(time)
			vecstims_bg_exc_apical[-1].delay = 0
			vecstims_bg_exc_apical[-1].play(events_bg_exc_apical[-1])
			syn_pos = rng_pos.repick()
			syn_bg_ampa_apical.append(h.GLU(sec(syn_pos)))
			syn_bg_nmda_apical.append(h.nmda(sec(syn_pos)))
			conn_bg_ampa_apical.append(h.NetCon(vecstims_bg_exc_apical[-1], syn_bg_ampa_apical[-1], -20, 0, ampa_g))
			conn_bg_nmda_apical.append(h.NetCon(vecstims_bg_exc_apical[-1], syn_bg_nmda_apical[-1], -20, 0, nmda_g))
	mprint("Allocated apical background excitatory synapses.")
	
else:
	for i,sec in enumerate(apical):
		bg_number_attention_driven_persection.append(int(apical_nsyn_bg_exc[i]*bg_fraction_attention_driven))
		bg_number_noise_driven_persection.append(apical_nsyn_bg_exc[i] - bg_number_attention_driven_persection[-1])
		
# Apical inhibitory background
if apical_bg_inh_enabled:
	events_bg_inh_apical = []
	vecstims_bg_inh_apical = []
	syn_bg_gaba_apical = []
	conn_bg_gaba_apical = []
	
	for i,sec in enumerate(apical):
		sec
		for isyn in range(0, apical_nsyn_bg_inh[i]):
			events_bg_inh_apical.append(h.Vector())
			vecstims_bg_inh_apical.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_inh.repick() > 0:
					events_bg_inh_apical[-1].append(time)
			vecstims_bg_inh_apical[-1].delay = 0
			vecstims_bg_inh_apical[-1].play(events_bg_inh_apical[-1])
			syn_pos = rng_pos.repick()
			syn_bg_gaba_apical.append(h.GABAa(sec(syn_pos)))
			conn_bg_gaba_apical.append(h.NetCon(vecstims_bg_inh_apical[-1], syn_bg_gaba_apical[-1], -20, 0, gaba_g))
	mprint("Allocated apical background inhibitory synapses.")
			

#%%=========================== Stim-driven synapses ================================

stim_fraction_attention_driven = afst         # Fraction of stimulus-driven synapses to be allocated to the "attentional" pool
stim_number_attention_driven_persection = []  # List to hold the number of synapses allocated to the "attentional" pool, per dendrite
stim_number_stim_driven_persection = []       # List to hold the remaining, stimulus-driven synapses

# Background excitation/inhibition parameters
freq_stim = stimulus_base_frequency           # Event frequency to be used in the Poisson spike generator(s)
stim_rng_offset = 50000                       # Offset number (fixed), to be used to alter random number seeds in predictable ways

# Toggle stim-driven excitation components on-off
basal_stim_enabled = bst
apical_stim_enabled = ast

# Emulation of different responses for each stimulus orientation via differentially weighted "tagged" synapses
presented_stimulus = stim_orient      # orientation of the stimulus actually being presented (value from 0 to 90, in 10 degree steps)
stims =  [x*10 for x in range(0,37)]  # possible orientations (0 to 360, steps of 10 degrees - but we focus mainly on 0 to 90)

# Weight vector of 36 values (37,but the 37th -360deg- wraps around to the 1st, 0deg) that correspond to all stimulus orientations
# The values correspond to activation frequencies for synapses that have an orientation preference matching each of the orientations here
activation_vector = [0.9974,0.2487,0.0039,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0039,0.2487,0.9974,0.2487,0.0039,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0039,0.2487,0.9974]
activation_vector = [x*stim_factor for x in activation_vector]  # Rescale activation frequencies to increase firing rate and reduce the dependence on multiple runs


def gauss5pdf(width, x, mean):
	"""Function that describes a wrapped Gaussian distribution.
	It features two peaks, which are 180deg apart. The peaks lie at <mean> and <mean+180> degrees, and the distribution is <width> degrees wide (st.deviation)"""
	return (1/(width*np.sqrt(2*np.pi))) * np.sum( [ np.exp(-0.5 * ((x-mean+2*k*180)**2)/(2*(width**2)) ) for k in [-1,-0.5,0,0.5,1] ] )		


#%%=====================  RNG (Random Number Generator) setup ===============================
		
# Synapses are "tagged" with an orientation preference (i.e. fire most frequently for that stimulus orientation) via probabilistic sampling
# so we need to initialize some random number generators to set up this sampling process.
# Pay attention to the generator used to ensure each individual basal tree section
# has the same mean orientation preference (to avoid extreme outliers), 'rng_basal_pref_mean'.
np.random.seed(int(rng_seed2))
rng_t_stim = h.Random(rng_seed1)  # This random number generator will be used to generate event timings ('_t_stim').
# We don't instantiate rng_t_stim as a poisson generator yet, as we need to include a frequency value from activation_vec, which relies on
# allocation of synaptic tags (orientation preferences)

# RNG for sampling mean orientation preference of a specific dendrite (in case there is a disparity between apical and basal mean orientation preferences)
rng_pref_mean = h.Random(rng_seed2-stim_rng_offset)
rng_pref_mean.discunif(int(tag_basal/10)-int(tag_apical/10),int(tag_basal/10)+int(tag_apical/10))

# RNG for determining the position of synapes on each individual dendrite
# Remember the notation used by NEURON:
# Section(0) refers to the origin/root of 'Section', and Section(1) refers to its distal terminal.
# Other (floating-point) numbers 'x' between 0 and 1 can be used, refering to the nearest existing segment
# on that compartment, which is (x*100)% of the distance from the root towards the distal terminal.
# Thus, Section(0.5) refers to a point in the middle of the Section, and Section(0.33) refers to a point
# 33% of the distance from the root to the distal terminal of the Section (or exactly on its first third).
rng_pos_stim = h.Random(rng_seed2+stim_rng_offset)
rng_pos_stim.uniform(0,1)


#%%=====================  Stim-driven synapse allocation  ===============================

bsc = 3   # Scales frequency of synapses preferring orientations close to 90/270 deg (orthogonal to intended preference)

# Allocate basal stim-driven synapses
# The comments here also apply to all other instances of stimulus-driven synapses
if basal_stim_enabled:
	tags_basal = []            # List of individual synaptic orientation preferences (after sampled from the distribution)
	events_stim_basal = []     # List of presynaptic events
	vecstims_stim_basal = []   # List of synaptic activation events (spike trains) - shared between AMPA and NMDA synapses
	syn_stim_basal_ampa = []        # List of AMPA synaptic objects
	syn_stim_basal_nmda = []        # List of NMDA synaptic objects
	conn_stim_basal_ampa = []       # List of connection objects for AMPA synapses
	conn_stim_basal_nmda = []       # List of connection objects for NMDA synapses
	debug_stim_basal = []           # Debugging tool - logs the properties of individual synapses during allocation
	
	for i,sec in enumerate(basal):        # For every basal dendrite...
		target_mean_preference = rng_pref_mean.repick()  # Select a mean orientation preference for the entire dendrite (typically 0 deg)
		if target_mean_preference < 0:     # If the chosen preference is somehow negative...
			target_mean_preference += 180  # ...project it onto the other half of the unit circle
		per_orient_probs = [gauss5pdf(basal_width,x,target_mean_preference) for x in stims]  # Create a wrapped gaussian distribution of synaptic orientation preferences
		per_orient_norm_probs = [x/np.sum(per_orient_probs) for x in per_orient_probs]  # Convert the above distribution into probability of allocation (normalize between 0-1)
		syn_tags = np.random.choice(stims, basal_nsyn_stim[i], p=per_orient_norm_probs) # Randomly sample 'basal_nsyn_stims[i]'' (number of synapses onto this dendrite) synaptic orientation preferences
		tags_basal.append(list(syn_tags))  # Append the sampled synaptic orientation preferences onto the appropriate list
		for j,syn in enumerate(syn_tags):  # For each synapse that this dendrite is supposed to have...
			if syn not in [70, 80, 90, 100, 110] and syn not in [250, 260, 270, 280, 290]:  # If the synaptic orientation preference is not orthogonal to 0 or 180, plus/minus 20 degrees...
				distance_from_tag = np.abs([x-np.abs(syn-presented_stimulus) for x in stims]) # Compute the angular distance of the synaptic orientation preference from the stimulus orientation
				activation_idx = np.argmin(distance_from_tag) # Use the calculated distance to find the index of the appropriate activation frequency for a synapse with this orientation preference
				rng_t_stim.poisson((freq_stim*activation_vector[activation_idx])/1000) # Create a Poisson spike generator with the appropriate activation frequency
			else:                                                                           # ...else, if the synaptic orientation preference doesn't match the above criteria...
				fct = 2 if (syn == 90 or syn == 270) else 1  # extra scaling factor for orthogonally-tuned synapses
				bad_synapse_coeff = bsc*fct                  # rescaling of activation frequency for orthogonally-tuned synapses
				distance_from_tag = np.abs([x-np.abs(syn-presented_stimulus) for x in stims]) # Compute the angular distance of the synaptic orientation preference from the stimulus orientation
				activation_idx = np.argmin(distance_from_tag) # Use the calculated distance to find the index of the appropriate activation frequency for a synapse with this orientation preference
				rng_t_stim.poisson((freq_stim*activation_vector[activation_idx]*bad_synapse_coeff)/1000) # Create a Poisson spike generator with the appropriate activation frequency, plus some scaling
			debug_stim_basal.append((j,syn,activation_idx,activation_vector[activation_idx])) # If needed, write the parameters of this synapse in the log file
			
			events_stim_basal.append(h.Vector())     # Append an empty Vector() (NEURON list-like object) onto the presynaptic events list, to be filled in shortly 
			vecstims_stim_basal.append(h.VecStim())  # Append an empty VecStim() (NEURON custom spike train playback object) onto the spike trains list, to be filled in shortly
			for time in range(0, int(h.tstop)):      # For each timepoint of the simulation (not ms, timepoint, determined by the h.dt parameter - 0.1 ms)...
				if rng_t_stim.repick() > 0:          # ...if the RNG of event timings returns a value over 0...
					events_stim_basal[-1].append(time)  #... then we have an event, and the current simulation timepoint is recorded in the previously created Vector() object
			vecstims_stim_basal[-1].delay = 500         # Now we set up the VecStim() object - it has an initial delay of 500 ms (delay before stimulus presentation)
			vecstims_stim_basal[-1].play(events_stim_basal[-1])  # Here, we link the presynaptic events in the Vector() to the VecStim(), so they can be "played back"
			syn_pos = rng_pos_stim.repick()                      # Pick a new random position on the dendrite to place this new synapse in
			syn_stim_basal_ampa.append(h.GLU(sec(syn_pos)))      # Create a new AMPA synapse on this position
			syn_stim_basal_nmda.append(h.nmda(sec(syn_pos)))     # ... and also create a new NMDA synapse on the same position
			conn_stim_basal_ampa.append(h.NetCon(vecstims_stim_basal[-1], syn_stim_basal_ampa[-1], -20, 0, ampa_g))  # Link the AMPA synapse with the VecStim() object, and set up synaptic properties
			conn_stim_basal_nmda.append(h.NetCon(vecstims_stim_basal[-1], syn_stim_basal_nmda[-1], -20, 0, nmda_g))  # Link the NMDA synapse with the VecStim() object, and set up synaptic properties
	mprint("Allocated basal stim-driven synapses.")
	log('='*30)  # Separator, to aid readability of the log
	for entry in debug_stim_basal:
		log(entry) # Log all entries in the debug list


# Allocate apical stim-driven synapses
if apical_stim_enabled:
	tags_apical = []
	events_stim_apical = []
	vecstims_stim_apical = []
	syn_stim_apical_ampa = []
	syn_stim_apical_nmda = []
	conn_stim_apical_ampa = []
	conn_stim_apical_nmda = []
	
	for i,sec in enumerate(apical):
		stim_number_attention_driven_persection.append(int(apical_nsyn_stim[i]*stim_fraction_attention_driven))
		stim_number_stim_driven_persection.append(apical_nsyn_stim[i] - stim_number_attention_driven_persection[-1])
		mprint(f'\tApical {i} stimulus-driven synapses ({apical_nsyn_stim[i]}) split: {stim_number_attention_driven_persection[-1]} attention-driven, {stim_number_stim_driven_persection[-1]} stimulus-driven.')
		
		per_orient_probs = [gauss5pdf(apical_width,x,tag_apical) for x in stims]
		per_orient_norm_probs = [x/np.sum(per_orient_probs) for x in per_orient_probs]  # normalization of probabilities
		#syn_tags = np.random.choice(stims, apical_nsyn_stim[i], p=per_orient_norm_probs)
		syn_tags = np.random.choice(stims, stim_number_stim_driven_persection[-1], p=per_orient_norm_probs) # for attention/stim split
		tags_apical.append(list(syn_tags))
		for syn in syn_tags:
			if syn not in [70, 80, 90, 100, 110] and syn not in [250, 260, 270, 280, 290]:
				distance_from_tag = np.abs([x-np.abs(syn-presented_stimulus) for x in stims])
				activation_idx = np.argmin(distance_from_tag)
				rng_t_stim.poisson((freq_stim*activation_vector[activation_idx])/1000)
			else:
				fct = 2 if (syn == 90 or syn == 270) else 1  
				bad_synapse_coeff = bsc*fct                  
				distance_from_tag = np.abs([x-np.abs(syn-presented_stimulus) for x in stims])
				activation_idx = np.argmin(distance_from_tag)
				rng_t_stim.poisson((freq_stim*activation_vector[activation_idx]*bad_synapse_coeff)/1000)
			
			events_stim_apical.append(h.Vector())
			vecstims_stim_apical.append(h.VecStim())
			for time in range(0, int(h.tstop)):
				if rng_t_stim.repick() > 0:
					events_stim_apical[-1].append(time)
			vecstims_stim_apical[-1].delay = 500
			vecstims_stim_apical[-1].play(events_stim_apical[-1])
			syn_pos = rng_pos_stim.repick()
			syn_stim_apical_ampa.append(h.GLU(sec(syn_pos)))
			syn_stim_apical_nmda.append(h.nmda(sec(syn_pos)))
			conn_stim_apical_ampa.append(h.NetCon(vecstims_stim_apical[-1], syn_stim_apical_ampa[-1], -20, 0, ampa_g))
			conn_stim_apical_nmda.append(h.NetCon(vecstims_stim_apical[-1], syn_stim_apical_nmda[-1], -20, 0, nmda_g))
	mprint("Allocated apical stim-driven synapses.")
	
else: # If apical stim-driven inputs are not enabled, but attentional inputs are, then we still need to calculate the appropriate synaptic amounts
	for i,sec in enumerate(apical):
		stim_number_attention_driven_persection.append(int(apical_nsyn_stim[i]*stim_fraction_attention_driven))
		stim_number_stim_driven_persection.append(apical_nsyn_stim[i] - stim_number_attention_driven_persection[-1])
		

#%%====================== Allocate attention-driven apical synapses ===========================
# This exists only in case you want to "play around" with it.
# In the default model state, no synapses (zero) are explicitly allocated to this attentional "pool" of synapses,
# so this code essentially doesn't execute. The role of attention is taken up by background (noise) synapses,
# which act in a similar manner, increasing baseline voltage enough so that action potentials are generated
# more easily. 
# If you want to, you can try to change the behavior of the neuron by tweaking the parameters of attention
# and attentional inputs. But you don't have to!


attention_enabled = att

if attention_enabled:
	
	# Attention parameters
	freq_att = afreq
	att_offset_seed = 15  # RNG seed offset is used to disentangle the randomness of different synaptic types - otherwise, synapses that use the same RNG would end up with identical parameters.
	
	# Random number generators for position and event timings	
	rng_pos_att = h.Random(rng_seed2-att_offset_seed)
	rng_pos_att.uniform(0,1)
	rng_t_att = h.Random(rng_seed1+att_offset_seed)
	rng_t_att.poisson(freq_att/1000) # Attentional synapses are not stimulus-driven, they are a "binary" (on/off) signal that acts similarly to background-driven inputs
	                                 # Thus, the frequency of the Poisson spike generator is not modulated by the stimulus orientation, and the generator can be created here.

	events_att_apical = []
	vecstims_att_apical = []
	syn_att_ampa_apical = []
	syn_att_nmda_apical = []
	conn_att_ampa_apical = []
	conn_att_nmda_apical = []
	
	
	for i,sec in enumerate(apical):
		sec
		for isyn in range(0, stim_number_attention_driven_persection[i]+bg_number_attention_driven_persection[i]):
			events_att_apical.append(h.Vector())
			vecstims_att_apical.append(h.VecStim())
			debug = []
			for time in range(0, int(h.tstop)):
				if rng_t_att.repick() > 0:
					events_att_apical[-1].append(time)

			vecstims_att_apical[-1].delay = 0
			vecstims_att_apical[-1].play(events_att_apical[-1])
			syn_pos = rng_pos_att.repick()
			syn_att_ampa_apical.append(h.GLU(sec(syn_pos)))
			syn_att_nmda_apical.append(h.nmda(sec(syn_pos)))
			conn_att_ampa_apical.append(h.NetCon(vecstims_att_apical[-1], syn_att_ampa_apical[-1], -20, 0, ampa_g))
			conn_att_nmda_apical.append(h.NetCon(vecstims_att_apical[-1], syn_att_nmda_apical[-1], -20, 0, nmda_g))
	mprint("Allocated apical attention-driven synapses.")

		
#%%=============================== Run & Visualize =====================================	

soma_v = h.Vector()  # set up a recording vector for somatic voltage
soma_v.record(soma(0.5)._ref_v)  # record voltage at the middle (0.5) of the soma
t = h.Vector()      # set up a vector to record simulation time
t.record(h._ref_t)  # record time

if not _multiple_runs:
	h.addgraph_2("soma.v(0.5)", 0,sim_time, -75, 50)  # From 'hoc.files/basic-graphics.hoc' - disabled for multiple runs
	                                                  # Displays somatic voltage at the midpoint DURING simulation (real-time)!

# Run simulation
h.v_init = -79  # set starting voltage - all compartments have their currents set to equilibrium according to this voltage in 'current-balance.hoc'
h.run()  # Start the simulation

# Save output data (somatic voltage and simulation time)
if _pickle_save:
	util.pickle_dump((t, soma_v), f'results/n{n_neuron}_r{n_run}_s{presented_stimulus}_a{att}.pkl')
else:
	with open(f'results/n{n_neuron}_r{n_run}_s{presented_stimulus}_a{att}_somaV.dat', 'w') as f:
		for time,voltage in zip(t,soma_v):
			f.write(f'{time:.01f}\t{voltage}\n')

print(f'Done with n{n_neuron}_r{n_run}_s{presented_stimulus}_a{att}.')

if not _multiple_runs:
	input()  # Wait for input after a single simulation ends (to prevent the simulator quitting before the full voltage trace can be seen)