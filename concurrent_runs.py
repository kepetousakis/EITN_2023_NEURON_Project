# -*- coding: utf-8 -*-
"""
Created on Wed Sep 6 17:41:21 2023

Code to parallelize simulation execution (one simulation per CPU thread).

@author: KEPetousakis
"""

import concurrent.futures
from time import sleep
import numpy as np
import os
import psutil 

# Here, you determine how many different parameter combinations you will have
# Number of combinations is <number of neurons> x <number of runs> x <number of stims>,
# so be careful not to have too many, or it'll take forever to run!
# But if you have too few, you might randomly have a "bad" sample, which misrepresents
# the true picture...
# The "ideal" is to check whether results look promising with a few runs, then run more
# once you think you've identified a promising model configuration.
neurons = [int(x) for x in range(0,6)]          # "Ideal" number is 10 (range(0,10)), recommend testing with range(0,2) or range(0,3) (N=2-3)
runs = [int(x) for x in range(0,6)]             # "Ideal" number is 10 (range(0,10)), recommend testing with range(0,2) or range(0,3) (N=2-3)
# stims = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] # Ideally, you want all orientations from 0 to 90 in steps of 10 (only steps of 10!) (N=10)
stims = [0, 10, 20, 30, 60, 70, 80, 90]         # ...but you can also use this reduced setup, to limit the number of simulations (N=8)

# Number of simulations in total will be <neurons> x <runs> x <stims>, so even in the example cases, your total number of simulations
# will be anywhere from 32 to 90! Do take the number of simulations into account, and also keep in mind that each simulation will be slower
# than one you run on its own.

print(f'Submitting {len(neurons)*len(runs)*len(stims):.0f} simulations to the CPU', end='')

attn = 0   # Toggles attention on-off - only use this if you want to "play around" with attentional inputs!
batch = 1  # Don't change this - it tells the simulation that it's running in "batch" mode, so that it doesn't display GUI elements etc.


input_data = []

# This determines the number of threads to be used for concurrent simulations.
# If you want to still be able to use your computer while the simulations are running, I'd suggest toggling the "min_available_threads" setting
# from 0 to 1 or 2 (depending on how many threads your CPU has).
min_available_threads = 0
nproc = np.max([2, int(len(psutil.Process().cpu_affinity()))-min_available_threads])  

print(f' on {nproc:.0f} threads.')

simulation_file_name = 'v1_pcell_baseline.py'  # Make sure to change this parameter if you want to change the file being executed (using the same arguments).
if '.py' not in simulation_file_name:
	simulation_file_name+='.py'


for n in neurons:
	for r in runs:
		for s in stims:
			input_data.append((n,r,s,attn,batch))

def task(data):
	nrn = data[0]; run = data[1]; stim = data[2]; att = data[3]; bat = data[4]
	if (not os.path.exists(f'results/n{nrn}_r{run}_s{stim}_a{att}.pkl') and (not os.path.exists(f'results/n{nrn}_r{run}_s{stim}_a{att}_somaV.dat'))):
		os.system(f'python {simulation_file_name} nrn={nrn:.0f} run={run:.0f} stim={stim:.0f} att={att:.0f} batch={bat:.0f}')
	else:
		print(f'Skipping "python {simulation_file_name} nrn={nrn:.0f} run={run:.0f} stim={stim:.0f} att={att:.0f} batch={bat:.0f}" - found results at "results/n{nrn}_r{run}_s{stim}_a{att}..."')
	return f'Completed nrn={nrn:.0f} run={run:.0f} stim={stim:.0f} att={att:.0f} batch={bat:.0f}'

def main():
	with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
		tasklist_dict = {executor.submit(task, x): x for x in input_data}
		for future in concurrent.futures.as_completed(tasklist_dict):
			datum = tasklist_dict[future]
			try:
				status = future.result()
			except Exception as e:
				print(f'Exception {e} due to input {datum}')
			else:
				print(f'Input {datum} returned with status {status}')
				
if __name__ == '__main__':
	main()

