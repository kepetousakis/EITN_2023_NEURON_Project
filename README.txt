Code was created and tested on Windows 10 (64bit), and was also tested on the latest Ubuntu version.
In case of any issues, contact me ("Konstantinos-Evangelos Petousakis", aka "Kostas"). 
My e-mail address is kepetousakis[at]gmail.com, in case you need to contact me via e-mail.
Send me an e-mail so I can add you to the project Slack group ("EITN_2023_nrnproject")!


INSTRUCTIONS:

> You should already have Anaconda Python installed, alongside the NEURON package for Python.
	- You will also need to be running your NEURON/Python code through a Conda environment that has access to the NEURON package.
	  The (base) environment will be enough if you have properly installed the NEURON package.

> Compile the .mod files in the "mod.files" folder. 
    - https://www.neuron.yale.edu/neuron/static/docs/nmodl/unix.html   for unix-based OS      (TL;DR: execute "nrnivmodl" from inside "./mod.files")
    - https://www.neuron.yale.edu/neuron/static/docs/nmodl/mswin.html  for windows-based OS   (TL;DR: execute "mknrndll" from inside "./mod.files")

    - This folder contains all the files describing the voltage-gated channels and synaptic mechanisms in use by the model.
    - You should not need to tweak anything concerning voltage-gated channels.
    - To add new synapses, you should follow the example of existing synapses in the model (code is documented).

Syntax (python/NEURON pseudocode):
	===========================================================================================================================
  section_name                                  # to access a specific section (the target of your synapses)
  my_synapse_ampa = h.GLU(synapse_position)     # allocates a single AMPA synapse at "synapse_position" (0-1)
  my_synapse_nmda = h.nmda(synapse_position)    # allocates a single NMDA synapse at "synapse_position" (0-1)
  my_synapse_gabaa= h.GABAa(synapse_position)   # allocates a single GABAa synapse at "synapse_position" (0-1)
  my_vecstim = h.VecStim()                      # creates a VecStim object
	  my_vecstim.delay = 0                          # no delay to event playback (they start when simulation starts)
	  # my_vecstim.delay = 100                      # OR events start 100ms after simulation starts
	  my_vecstim.play(spike_train)                  # add events to the VecStim to play back ('spike_train' contains
	  												# the timing of events in ms, and can be created in many ways)
	  synaptic_threshold=-20                        # If spike_train[x] > synaptic_threshold, the synapse is activated
	  synaptic_delay=0                              # Synapses should not have an activation delay (ms)
	  synaptic_weight=0.001                         # Conductance of synaptic mechanisms - can have separate vars, of course

	  # Connect synapses with (in this case) the same event source (VecStim object) - can also create different sources
  my_connection_ampa = h.NetCon(my_vecstim, my_synapse_ampa, synaptic_threshold, synaptic_delay, synaptic_weight)
  my_connection_nmda = h.NetCon(my_vecstim, my_synapse_nmda, synaptic_threshold, synaptic_delay, synaptic_weight*1.1)
  my_connection_gabaa = h.NetCon(my_vecstim, my_synapse_nmda, synaptic_threshold, synaptic_delay, synaptic_weight*1.2)
===========================================================================================================================

> The "hoc.files" folder contains files that generate the reconstructed neuronal morphology and also allocate all non-synaptic 
  mechanisms required.
	- You can browse the files therein if you want, but they are written in pure NEURON, not NEURON/Python, so be warned!
	- You should not have to tweak the files in the "hoc.files" folder in any way. If you think you should, let me know!

> Verify that your copy of the model functions as intended by running the test model file (v1_pcell_test.py). 
	- The code in the test file already contains a simple functionality test with a dual pulse current injection, intended to generate
	  *a single action potential*, shown in a simple plot.
	- This verification script is not required to continue the project. You will work with "v1_pcell_baseline.py".
	- Ensure "v1_pcell_baseline.py" also works as intended by running it, first with "_test_function" True, and then False. 
	- It might look like the simulation has frozen when it first starts up (the GUI will show time not progressing), but that is normal.
	  It takes some time for the model to be prepared, since thousands of synapses need to be created and allocated before the simulation
	  can start.
	- If "v1_pcell_baseline.py" works properly, you should get a single action potential when "_test_function" is True, and
	  four action potentials when "_test_function" is False, and the "nrn" and "run" parameters have not been changed (default value: 0).
	- If that's not what you see, then try and figure out what the issue is! I'm always available for questions, if you have any.

> By default, the simulation will output results (simulation time + somatic voltage at the midpoint) in the "results" folder, with a
  file name determined by the simulation parameters (neuron ID, simulation ID, stimulus orientation, and whether attentional inputs
  are toggled on or not - but you don't have to worry about those, if you don't want to "play around" with them).

> The simulation also outputs most printed output in a log file, found in the "logs" folder.

> 'concurrent_runs.py' and 'process_results_get_OTCurves.py' are provided for your convenience. The former allows you to run multiple
  simulations at the same time, limited by the number of CPU threads available. The latter allows you to quickly parse saved simulation
  results and produce an orientation tuning curve, alongside the OSI. You do not have to use these files - especially the latter - but
  they can save you a lot of time, if you think you will need it.

> Move towards achieving the goals of the project as stated:
	- Main Question: "Can dendritic inhibition help cause the emergence of orientation tuning in a L2/3 V1 neuron?"

	Milestones:
	- Test model functionality - is it working properly?
	- Allocate a single new synapse and ensure correct function
	- Produce an orientation tuning curve (you will need multiple different stimulus orientations), and characterize it with the OSI
	- Allocate multiple (inhibitory?) synapses according to a set plan 
	- Investigate the effect of inhibition on neuronal output - can tuning emerge?
	- Is there something else you'd like to check using the model? You may do so at your discretion!
	- Present your results!


Useful Papers:
1. Silver, R. A. (2010). Neuronal arithmetic. Nature Reviews Neuroscience, 11(7), 474-489.
2. Goetz, L., Roth, A., & Häusser, M. (2021). Active dendrites enable strong but sparse inputs to determine orientation selectivity. Proceedings of the National Academy of Sciences, 118(30), e2017339118.
3. Gidon, A., & Segev, I. (2012). Principles governing the operation of synaptic inhibition in dendrites. Neuron, 75(2), 330-341.
4. Park, J., Papoutsi, A., Ash, R. T., Marin, M. A., Poirazi, P., & Smirnakis, S. M. (2019). Contribution of apical and basal dendrites to orientation encoding in mouse V1 L2/3 pyramidal neurons. Nature Communications, 10(1), 1-11.


Other Resources:
The project presentation (included as .pdf, and also as .pptx with speaker notes included)
The NEURON tutorials
https://neuron.yale.edu/neuron/static/py_doc/index.html  [NEURON/Python documentation]
https://docs.python.org/3/reference/  [Python documentation]