# Thalamo-cortical spiking model with sleep-like slow oscillations improving awake visual classification through synaptic homeostasis and memory association

The occurrence of sleep passed through the evolutionary sieve and is widespread in animal species. Sleep is known to be beneficial to cognitive and mnemonic tasks, while chronic sleep deprivation is detrimental. Despite the importance of the phenomenon, a complete understanding of its functions and underlying mechanisms is still lacking. In this paper, we show interesting effects of deep-sleep-like slow oscillation activity on a simplified thalamo-cortical model which is trained to encode, retrieve and classify images of handwritten digits. During slow oscillations, spike-timing-dependent-plasticity (STDP) produces a differential homeostatic process. It is characterized by both a specific unsupervised enhancement of connections among groups of neurons associated to instances of the same class (digit) and a simultaneous down-regulation of stronger synapses created by the training. This hierarchicalorganization of post-sleep internal representations favours higher performances in retrieval and classification tasks. The mechanism is based on the interaction between top-down cortico-thalamic predictions and bottom-up thalamo-cortical projections during deep-sleep-like slow oscillations. Indeed, when learned patterns are replayed during sleep, cortico-thalamo-cortical connections favour the activation of other neurons coding for similar thalamic inputs, promoting their association. Such mechanism hints at possible applications to artificial learning systems.


## Reference

C. Capone, E. Pastorelli, B. Golosio and P. S. Paolucci. Sleep-like slow oscillations improve visual classification through synaptic homeostasis and memory association in a thalamo-cortical model. Sci Rep 9, 8990 (2019) doi:10.1038/s41598-019-45525-0


## How to run

- Download the code and the input data ("mnist_preprocessing").
- Run each python file contained in the "Code" folder using the instruction:
  > python file_name
- The results are produced in the Code folder

NOTE: the code is executed using the NEST simulator. Having it installed is a prerequisite.
The code hase been tested on NEST v2.12
