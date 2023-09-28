# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:54:17 2023

Code to perform analysis of simulation results, and output a tuning curve and the corresponding
Orientation Selectivity Index (OSI).

@author: KEPetousakis
"""

import numpy as np
import matplotlib.pyplot as plt
import utility_spikes_pickling as util
import os
import warnings

warnings.filterwarnings("ignore")  # suppresses warnings related to graphing and functions close to being deprecated


# The parameters here need to match the ones in "concurrent_runs.py" - otherwise, some results might be skipped, or the 
# script might raise an exception.
neurons = [int(x) for x in range(0,6)]          # "Ideal" number is 10 (range(0,10)), recommend testing with range(0,2) or range(0,3) (N=2-3)
runs = [int(x) for x in range(0,6)]             # Ideal number is 10 (range(0,10)), recommend testing with range(0,2) or range(0,3) (N=2-3)
# stims = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] # Ideally, you want all orientations from 0 to 90 in steps of 10 (only steps of 10!) (N=10)
stims = [0, 10, 20, 30, 60, 70, 80, 90]         # ...but you can also use this reduced setup, to limit the number of simulations (N=8)
att = 0   # Toggles attention on-off - only use this if you want to "play around" with attentional inputs!


# For a symmetrical visualization of the tuning curve, toggle "symmetrical_view" to True (otherwise it's one-sided)
symmetrical_view = True

path_to_results = 'results'

_save_analysis_results = True
_analysis_results_filename = 'analysis_results.pkl'


data = np.zeros(shape=(len(neurons), len(runs), len(stims), 25001))
spikes = np.zeros(shape=(len(neurons), len(runs), len(stims)))

time = []

def load_data(n,r,s,a,type='.dat'):
	if type == '.pkl':
		(times, voltages) = util.pickle_load(f'./{path_to_results}//n{n}_r{r}_s{s}_a{a}.pkl')
	elif type == '.dat':
		data = np.loadtxt(f'./{path_to_results}//n{n}_r{r}_s{s}_a{a}_somaV.dat')
		data_shape = np.shape(data)
		times = data[:,0]
		voltages = data[:,1]
	else:
		raise Exception('Bad arguments to load_data()')
	return (times, voltages)

if not os.path.exists(f"./results/{_analysis_results_filename}"):

	for neuron in neurons:
		for run in runs:
			for istim,stim in enumerate(stims):
				print(f'Processing "{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}"...', end='')
				if os.path.exists(f'{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}.pkl'):
					(t,datum) = load_data(neuron,run,stim,att,type='.pkl')
				elif os.path.exists(f'{path_to_results}/n{neuron}_r{run}_s{stim}_a{att}_somaV.dat'):
					(t,datum) = load_data(neuron,run,stim,att,type='.dat')
				else:
					print(f'Error! Missing file: "{path_to_results}/n{neuron}_r{run}_s{stim}_a{att} ..."')
					raise Exception
				data[neuron, run, istim, :] = datum
				(nsp, times) = util.GetSpikes(datum[5000:])
				spikes[neuron, run, istim] = nsp/2
				print(f'found {nsp} spike(s) in 2s ({nsp/2:.2f} Hz)...', end='')
				if time == []:
					time = datum[0]
				print('done.')

	if _save_analysis_results:
		print(f'Saving analysis results in "./results/{_analysis_results_filename}".')
		util.pickle_dump(spikes, f'./results/{_analysis_results_filename}')

else:
	print(f'Loading analysis results from  "./results/{_analysis_results_filename}".')
	spikes = util.pickle_load(f"./results/{_analysis_results_filename}")

if symmetrical_view:
	stims_reflected = stims[1:]
	stims_reflected = [-x for x in stims_reflected]
	stims_reflected.reverse()
	stims_temp = stims_reflected + stims
	stims = stims_temp[:]
				
# Average across runs per neuron
spikes = np.mean(spikes, axis=1)

# Average across neurons, keep std
spikes_avg = np.mean(spikes, axis=0)
spikes_std = np.std(spikes, axis=0)
spikes_serr = np.std(spikes, axis=0)/np.sqrt(len(neurons))

# For a symmetrical visualization of the tuning curve (otherwise it's one-sided)
if symmetrical_view:

	stims_reflected = stims[1:]
	stims_reflected = [-x for x in stims_reflected]
	stims_reflected.reverse()
	stims_all = stims_reflected + stims

	r_pref = spikes_avg[0]; r_orth = spikes_avg[-1]

	spikes_avg_refl = [x for x in spikes_avg[1:]]; spikes_avg_refl.reverse()
	spikes_temp = [x for x in spikes_avg_refl] + [x for x in spikes_avg]
	spikes_avg = spikes_temp[:]
	
	spikes_std_refl = [x for x in spikes_std[1:]]; spikes_std_refl.reverse()
	spikes_std_temp = [x for x in spikes_std_refl] + [x for x in spikes_std]
	spikes_std = spikes_std_temp[:]

	spikes_serr_refl = [x for x in spikes_serr[1:]]; spikes_serr_refl.reverse()
	spikes_serr_temp = [x for x in spikes_serr_refl] + [x for x in spikes_serr]
	spikes_serr = spikes_serr_temp[:]
else:
	r_pref = spikes_avg[0]; r_orth = spikes_avg[-1]

# Plot average + error (standard deviation or standard error)
plt.figure()
# plt.errorbar(stims, spikes_avg, spikes_std, capsize=8, c='k')  # Plots tuning curve with standard deviation bars
plt.errorbar(stims, spikes_avg, spikes_serr, capsize=8, c='k')  # Plots tuning curve with standard error bars
plt.xlabel('Stimulus orientation (deg)')
plt.xticks(stims, stims)
plt.ylabel('Neuronal response (Hz)')

fig = plt.gcf()
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

OSI = lambda r_pref,r_orth: (r_pref-r_orth)/(r_pref+r_orth)

this_osi = OSI(r_pref, r_orth)

# print(f'Analysis complete.\nPreferred orientation firing rate: {spikes_avg[0]:.2f} Hz, orthogonal orientation firing rate {spikes_avg[-1]:.2f} Hz | OSI for this configuration (att={att}): {this_osi:.04f}.')

# plt.title(f'Tuning Curve (N = {len(neurons)*len(runs)}, att={att}, OSI={this_osi:.04f}), Rpref={spikes_avg[0]:.2f} Hz , Rorth={spikes_avg[-1]:.2f} Hz')

print(f'Analysis complete.\nPreferred orientation firing rate: {r_pref:.2f} Hz, orthogonal orientation firing rate {r_orth:.2f} Hz | OSI for this configuration (att={att}): {this_osi:.04f}.')

plt.title(f'Tuning Curve (N = {len(neurons)*len(runs)}, att={att}, OSI={this_osi:.04f}), Rpref={r_pref:.2f} Hz , Rorth={r_orth:.2f} Hz')

plt.show()

input()
