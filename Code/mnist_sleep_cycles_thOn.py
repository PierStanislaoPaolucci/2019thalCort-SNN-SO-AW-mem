#####################################################################################
#
# mnist_sleep_cycles_thOn.py
# INFN - November 2018
#
# Training, Sleep and Classification on a Thalamo-cortical model used to generate 
# the data reported in Fig.5 of the following article:
#
# Capone, C., Pastorelli, E., Golosio, B. and Paolucci, P.S. 
# Sleep-like slow oscillations improve visual classification through synaptic 
# homeostasis and memory association in a thalamo-cortical model. Sci Rep 9, 8990 (2019)
#
# DOI: https://doi.org/10.1038/s41598-019-45525-0
#
#####################################################################################

import sys
import nest
import pylab
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio

N_CLASSES = 10
N_RANKS = 3
N_TEST = 250

N_cycles = 30

startTime = time.time()

fn_train = '../mnist_preprocessing/mnist_training_features_0.9_14.npy'
feat_arr_train0 = np.load(fn_train)
label_fn_train = '../mnist_preprocessing/mnist_training_labels_0.9_14.npy'
labels_train0 = np.load(label_fn_train)
fn_test = '../mnist_preprocessing/mnist_test_features_0.9_14.npy'
feat_arr_test = np.load(fn_test)
label_fn_test = '../mnist_preprocessing/mnist_test_labels_0.9_14.npy'
labels_test = np.load(label_fn_test)

feat_train_class=[[] for i in range(N_CLASSES)] # empty 2d list (N_CLASSES x N)
label_train_class=[[] for i in range(N_CLASSES)]
for i in range(len(labels_train0)):
    i_class = labels_train0[i]
    feat_train_class[i_class].append(feat_arr_train0[i])
    label_train_class[i_class].append(i_class)

seed_rand = 31
subset = seed_rand

feat_red = [feat_train_class[i][j] for i in range(N_CLASSES)
            for j in range(subset*N_RANKS,(subset+1)*N_RANKS)]
labels_red = [label_train_class[i][j] for i in range(N_CLASSES)
              for j in range(subset*N_RANKS,(subset+1)*N_RANKS)]

rand = np.random.RandomState(12345)

feat_arr_train = np.asarray(feat_red)
labels_train = np.asarray(labels_red)

labels_train=labels_train.astype(int)
labels_test=labels_test.astype(int)

msd = 1234506+subset
nest.SetKernelStatus({"local_num_threads": 32})
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
nest.SetKernelStatus({'grng_seed' : msd+N_vp})
nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})

#parameters
r_noise_inp = 30000.0
r_noise_out = 30000.0
r_noise_exc = 2000.0 
r_noise_inh = 10000.0
r_noise_inp_recog = r_noise_inp
r_noise_exc_sleep = 710.

W_noise_inp = 8.0
W_noise_exc = 15.0
W_noise_out = 1.4
W_noise_inh = 5.0

W0_exc_inp = 1.
W0_inp_exc = 1.
W0_exc_out = 0.0001
W0_exc_exc = 1.

Wmax_inp_exc = 5.50
Wmax_exc_inp = 130.
Wmax_exc_out = 154.0
Wmax_exc_exc = 150.

W_exc_inh = 60.0
W_inh_exc = -4.
W_inh_exc_retrieval = -4. 
W_inh_inh = -1.
W_inh_exc_sleep = -.5
W_inp_inh2 = 10.0
W_inh2_inp = -1.0
c_m_sleep = 281.

mu_minus = 1.
mu_plus =  1.
alpha = 1.
alpha_sleep = 20.
b_sleep = 120.

lambda_inp_exc = 0.03
lambda_exc_inp = .08
lambda_exc_out = .05
lambda_exc_exc = 0.12

lambda_inp_exc_sleep = 0.0
lambda_exc_out_sleep = 0.0
lambda_exc_exc_sleep = .000002

t_train = 400.0
t_pause = 400.0
t_check = 200.0
t_sleep = 100000.

input_size = 324
n_classes = N_CLASSES

train_digit=feat_arr_train
teach_out=labels_train
n_train = len(teach_out)

test_digit=feat_arr_test[0:N_TEST]
test_out=labels_test[0:N_TEST]
n_test = len(test_out)

n_inh = 200
k_exc = 20
n_exc = k_exc*n_train
n_inh2 = 200

######################################################################
n_inp = input_size
n_out = n_classes

# build input patterns
train_pattern = [[0 for i in range(n_inp)] for j in range(n_train)]
for i in range(input_size):
    for i_train in range(n_train):
        train_pattern[i_train][i] = train_digit[i_train][i]

test_pattern = [[0 for i in range(n_inp)] for j in range(n_test)]
for i in range(input_size):
    for i_test in range(n_test):
        test_pattern[i_test][i] = test_digit[i_test][i]

n_spikes_inp = [0] * n_inp
cum_spikes_inp = [0] * n_inp
n_spikes_exc = [0] * n_exc
cum_spikes_exc = [0] * n_exc
n_spikes_inh = [0] * n_inh
cum_spikes_inh = [0] * n_inh
n_spikes_out = [0] * n_out
cum_spikes_out = [0] * n_out
n_spikes_inh2 = [0] * n_inh2
cum_spikes_inh2 = [0] * n_inh2

neur_inp=nest.Create("aeif_cond_alpha", n_inp) # input neurons
neur_exc=nest.Create("aeif_cond_alpha", n_exc) # excitatory neurons
neur_inh=nest.Create("aeif_cond_alpha", n_inh) # inhibitory neurons
neur_out=nest.Create("aeif_cond_alpha", n_out)   # output neurons (classes)
neur_inh2=nest.Create("aeif_cond_alpha", n_inh2) # inhibitory neurons 2

nest.SetStatus(neur_inp, {"b": .01})
nest.SetStatus(neur_exc, {"b": .01})
nest.SetStatus(neur_inh, {"b": .01})
nest.SetStatus(neur_out, {"b": .01})
nest.SetStatus(neur_inh2, {"b": .01})

nest.SetStatus(neur_inp, {"t_ref": 2.0})
nest.SetStatus(neur_exc, {"t_ref": 2.0})
nest.SetStatus(neur_inh, {"t_ref": 2.0})
nest.SetStatus(neur_out, {"t_ref": 2.0})
nest.SetStatus(neur_inh2, {"t_ref": 2.0})

#random V to exc neurons

Vrest = -71.2
Vth = -70.

np.random.seed(seed_rand)
Vms = Vrest+(Vth-Vrest)*np.random.rand(len(neur_exc))
nest.SetStatus(neur_exc, "V_m", Vms)

# input noise
noise_inp = nest.Create("poisson_generator", n_inp)
syn_dict_noise_inp = {"weight": W_noise_inp, "delay":1.0}
nest.Connect(noise_inp, neur_inp, "one_to_one", syn_dict_noise_inp)

# training noise
noise_exc = nest.Create("poisson_generator",n_exc)
syn_dict_noise_exc = {"weight": W_noise_exc, "delay":3.0}
conn_dict_noise_exc = {'rule': 'one_to_one'}
nest.Connect(noise_exc, neur_exc, conn_dict_noise_exc, syn_dict_noise_exc)

# build train exc neuron groups
train_target_exc = [[0 for i in range(k_exc)] for j in range(n_train)]
tgt=range(n_exc)
for i_train in range(n_train):
    for i_k in range(k_exc):
        i = i_train*k_exc + i_k
        train_target_exc[i_train][i_k]=i
        
 # teaching output noise
noise_out = nest.Create("poisson_generator",n_out)
syn_dict_noise_out = {"weight": W_noise_out, "delay":1.0}
nest.Connect(noise_out, neur_out, "one_to_one", syn_dict_noise_out)

# inhibitory noise
noise_inh = nest.Create("poisson_generator",n_inh)
syn_dict_noise_inh = {"weight": W_noise_inh, "delay":1.0}
nest.Connect(noise_inh, neur_inh, "one_to_one", syn_dict_noise_inh)

# spike detectors for input layer
sd_inp = nest.Create("spike_detector", n_inp)
nest.SetStatus(sd_inp, {"withgid": True, "withtime": True})
nest.Connect(neur_inp, sd_inp, "one_to_one")

# spike detectors for excitatory neurons
sd_exc = nest.Create("spike_detector", n_exc)
nest.SetStatus(sd_exc, {"withgid": True, "withtime": True})
nest.Connect(neur_exc, sd_exc, "one_to_one")

# spike detectors for inhibitory neurons
sd_inh = nest.Create("spike_detector", n_inh)
nest.SetStatus(sd_inh, {"withgid": True, "withtime": True})
nest.Connect(neur_inh, sd_inh, "one_to_one")

# spike detectors for output layer
sd_out = nest.Create("spike_detector", n_out)
nest.SetStatus(sd_out, {"withgid": True, "withtime": True})
nest.Connect(neur_out, sd_out, "one_to_one")

# spike detectors for inhibitory neurons 2
sd_inh2 = nest.Create("spike_detector", n_inh2)
nest.SetStatus(sd_inh2, {"withgid": True, "withtime": True})
nest.Connect(neur_inh2, sd_inh2, "one_to_one")

# input to excitatory connections
syn_dict_inp_exc = {"model": "stdp_synapse", "lambda": lambda_inp_exc, "weight":W0_inp_exc, "Wmax":Wmax_inp_exc, "delay":1.0}
conn_dict = {'rule': 'pairwise_bernoulli', 'p': 1.}
nest.Connect(neur_inp, neur_exc, conn_dict, syn_dict_inp_exc)

syn_dict_exc_inp = {"model": "stdp_synapse", "lambda": lambda_exc_inp, "weight":W0_exc_inp, "Wmax":Wmax_exc_inp, "delay":1.0}
nest.Connect(neur_exc, neur_inp, "all_to_all", syn_dict_exc_inp)

# excitatory to output connections
syn_dict_exc_out = {"model": "stdp_synapse", "lambda": lambda_exc_out, "weight":W0_exc_out, "Wmax":Wmax_exc_out, "delay":1.0}
nest.Connect(neur_exc, neur_out, "all_to_all", syn_dict_exc_out)

w_min = 0.001
w_max = W0_exc_exc
# excitatory to excitatory connections
syn_dict_exc_exc = {"model": "stdp_synapse", "lambda": lambda_exc_exc, "weight":W0_exc_exc, "Wmax":Wmax_exc_exc,"delay":
                    {'distribution': 'exponential_clipped',
                              'lambda': 10.,
                              'low': 1.,
                              'high': 10.},
                    "weight": {"distribution": "uniform", "low": w_min, "high": w_max}}
nest.Connect(neur_exc, neur_exc, "all_to_all", syn_dict_exc_exc)

# excitatory to inhibitory connections
syn_dict_exc_inh = {"weight": W_exc_inh, "delay":1.0}
nest.Connect(neur_exc, neur_inh, "all_to_all", syn_dict_exc_inh)

# inhibitory to excitatory connections
syn_dict_inh_exc = {"weight": W_inh_exc, "delay":1.0}
nest.Connect(neur_inh, neur_exc, "all_to_all", syn_dict_inh_exc)

# inh to inh
syn_dict_inh_inh = {"weight": W_inh_inh, "delay":1.0}
nest.Connect(neur_inh, neur_inh, "all_to_all", syn_dict_inh_exc)

# input to inhibitory connections
syn_dict_inp_inh2 = {"weight": W_inp_inh2, "delay":1.0}
nest.Connect(neur_inp, neur_inh2, "all_to_all", syn_dict_inp_inh2)

# inhibitory to input connections
syn_dict_inh2_inp = {"weight": W_inh2_inp, "delay":1.0}
nest.Connect(neur_inh2, neur_inp, "all_to_all", syn_dict_inh2_inp)


######################################################################
# training

nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_minus': mu_minus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'mu_minus': mu_minus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_minus': mu_minus})

nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'mu_plus': mu_plus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'mu_plus': mu_plus})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'mu_plus': mu_plus})

nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'alpha': alpha})
nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'alpha': alpha})

for i_train in range(n_train):
    i_out = teach_out[i_train]
    print "i_out label = ",  i_out
    print "Training number ", i_train

    Nup = 0
    for i in range(n_inp):
        if train_pattern[i_train][i]==1:
            nest.SetStatus([noise_inp[i]], {"rate": r_noise_inp})
            Nup = Nup + 1
        else:
            nest.SetStatus([noise_inp[i]], {"rate": 0.0})
    print "Nup ", Nup

    for i in range(n_exc):
        nest.SetStatus([noise_exc[i]], {"rate": 0.0})

    for i_k in range(k_exc):
            nest.SetStatus([noise_exc[train_target_exc[i_train][i_k]]],
                           {"rate": r_noise_exc  } )
  
    for i in range(n_out):
        if i==i_out:
            nest.SetStatus([noise_out[i]], {"rate": r_noise_out})
        else:
            nest.SetStatus([noise_out[i]], {"rate": 0.0})

    for i in range(n_inh):
        nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh})

    # unfreeze weights for training
    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc),
                   {'lambda': lambda_inp_exc})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_inp),
                   {'lambda': lambda_exc_inp})

    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out),
                   {'lambda': lambda_exc_out})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc),
                   {'lambda': lambda_exc_exc})

    #simulation
    nest.Simulate(t_train)
    # freeze weights for running tests
    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc),
                   {'lambda': 0.0})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_inp),
                   {'lambda': 0.0})

    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out),
                   {'lambda': 0.0})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc),
                   {'lambda': 0.0})

    evt_sd_inp = nest.GetStatus(sd_inp,keys="events")
    for i in range(n_inp):
        send_sd_inp = evt_sd_inp[i]["senders"]
        t_sd_inp = evt_sd_inp[i]["times"]
        n_spikes_inp[i] = len(t_sd_inp) - cum_spikes_inp[i];
        cum_spikes_inp[i] = len(t_sd_inp);
    evt_sd_out = nest.GetStatus(sd_out,keys="events")
    for i in range(n_out):
        send_sd_out = evt_sd_out[i]["senders"]
        t_sd_out = evt_sd_out[i]["times"]
        n_spikes_out[i] = len(t_sd_out) - cum_spikes_out[i];
        cum_spikes_out[i] = len(t_sd_out);
    evt_sd_exc = nest.GetStatus(sd_exc,keys="events")
    for i in range(n_exc):
        send_sd_exc = evt_sd_exc[i]["senders"]
        t_sd_exc = evt_sd_exc[i]["times"]
        n_spikes_exc[i] = len(t_sd_exc) - cum_spikes_exc[i];
        cum_spikes_exc[i] = len(t_sd_exc);
    evt_sd_inh = nest.GetStatus(sd_inh,keys="events")
    for i in range(n_inh):
        send_sd_inh = evt_sd_inh[i]["senders"]
        t_sd_inh = evt_sd_inh[i]["times"]
        n_spikes_inh[i] = len(t_sd_inh) - cum_spikes_inh[i];
        cum_spikes_inh[i] = len(t_sd_inh);
    evt_sd_inh2 = nest.GetStatus(sd_inh2,keys="events")
    for i in range(n_inh2):
        send_sd_inh2 = evt_sd_inh2[i]["senders"]
        t_sd_inh2 = evt_sd_inh2[i]["times"]
        n_spikes_inh2[i] = len(t_sd_inh2) - cum_spikes_inh2[i];
        cum_spikes_inh2[i] = len(t_sd_inh2);

    #switch off all teaching outputs
    for i in range(n_exc):
        nest.SetStatus([noise_exc[i]], {"rate": 0.0})

    for i in range(n_out):
        nest.SetStatus([noise_out[i]], {"rate": 0.0})

    for i in range(n_inh):
        nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh*4})

    for i in range(n_inp):
        nest.SetStatus([noise_inp[i]], {"rate": 0.0})

    nest.Simulate(t_pause)

accuracy_vector=np.zeros((N_cycles,1),dtype=float)
fr_vector=np.zeros((n_exc,N_TEST),dtype=float)

for n_cycle in range(N_cycles):

    ######################################################################
    #Test
    
    nest.SetStatus(nest.GetConnections(source=neur_inh, target=neur_exc), {"weight": W_inh_exc_retrieval})
    
    #switch off all teaching outputs
    for i in range(n_exc):
        nest.SetStatus([noise_exc[i]], {"rate": 0.0})
    
    for i in range(n_out):
        nest.SetStatus([noise_out[i]], {"rate": 0.0})
    
    for i in range(n_inh):
        nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh*4})
    
    for i in range(n_inp):
        nest.SetStatus([noise_inp[i]], {"rate": 0.0})
    
    nest.Simulate(t_pause*20)
    
    
    print '######################################################################'
    print '# Test'
    print '######################################################################'
    print
    
    #test_seq = [2,0,1,3]
    
    count_right = 0
    count_unsupervised_right =0
    for i_test in range(n_test):
    
        #switch off all teaching outputs
        for i in range(n_exc):
            nest.SetStatus([noise_exc[i]], {"rate": 0.0})
    
        for i in range(n_out):
            nest.SetStatus([noise_out[i]], {"rate": 0.0})
    
        for i in range(n_inh):
            nest.SetStatus([noise_inh[i]], {"rate": 0.0})
    
        #prepare input pattern
        Nup = 0
        for i in range(n_inp):
            if test_pattern[i_test][i]==1:
                nest.SetStatus([noise_inp[i]], {"rate": r_noise_inp_recog})
                Nup = Nup + 1
            else:
                nest.SetStatus([noise_inp[i]], {"rate": 0.0})
        print "Nup ", Nup
    
        nest.Simulate(t_check)
    
        evt_sd_inp = nest.GetStatus(sd_inp,keys="events")
        for i in range(n_inp):
            send_sd_inp = evt_sd_inp[i]["senders"]
            t_sd_inp = evt_sd_inp[i]["times"]
            n_spikes_inp[i] = len(t_sd_inp) - cum_spikes_inp[i];
            cum_spikes_inp[i] = len(t_sd_inp);
        evt_sd_out = nest.GetStatus(sd_out,keys="events")
        for i in range(n_out):
            send_sd_out = evt_sd_out[i]["senders"]
            t_sd_out = evt_sd_out[i]["times"]
            n_spikes_out[i] = len(t_sd_out) - cum_spikes_out[i];
            cum_spikes_out[i] = len(t_sd_out);
        evt_sd_exc = nest.GetStatus(sd_exc,keys="events")
        for i in range(n_exc):
            send_sd_exc = evt_sd_exc[i]["senders"]
            t_sd_exc = evt_sd_exc[i]["times"]
            n_spikes_exc[i] = len(t_sd_exc) - cum_spikes_exc[i];
            cum_spikes_exc[i] = len(t_sd_exc);
        evt_sd_inh = nest.GetStatus(sd_inh,keys="events")
        for i in range(n_inh):
            send_sd_inh = evt_sd_inh[i]["senders"]
            t_sd_inh = evt_sd_inh[i]["times"]
            n_spikes_inh[i] = len(t_sd_inh) - cum_spikes_inh[i];
            cum_spikes_inh[i] = len(t_sd_inh);
        evt_sd_inh2 = nest.GetStatus(sd_inh2,keys="events")
        for i in range(n_inh2):
            send_sd_inh2 = evt_sd_inh2[i]["senders"]
            t_sd_inh2 = evt_sd_inh2[i]["times"]
            n_spikes_inh2[i] = len(t_sd_inh2) - cum_spikes_inh2[i];
            cum_spikes_inh2[i] = len(t_sd_inh2);
    
        #switch off all teaching outputs
        for i in range(n_exc):
            nest.SetStatus([noise_exc[i]], {"rate": 0.0})
    
        for i in range(n_out):
            nest.SetStatus([noise_out[i]], {"rate": 0.0})
    
        for i in range(n_inh):
            nest.SetStatus([noise_inh[i]], {"rate": r_noise_inh*2})
    
        for i in range(n_inp):
            nest.SetStatus([noise_inp[i]], {"rate": 0.0})

        nest.Simulate(t_pause)
    
        n_spikes_max = 0
        i_out = -1
        for i in range(n_out):
            n_spikes = n_spikes_out[i];
            if n_spikes>n_spikes_max:
                i_out = i
                n_spikes_max = n_spikes

        print 'Output class index: ', i_out
        print 'Target class index: ', test_out[i_test]
        if i_out==test_out[i_test]:
            count_right = count_right + 1
        print 'Corrects: ', count_right, '/', i_test+1
        print 'Accuracy = ', float(count_right)/float(i_test+1.)*100., '%'
    
        n_spikes_max_exc = 0
        i_out_exc = -1
        for i in range(n_exc):
            n_spikes = n_spikes_exc[i];
            fr_vector[i,i_test] = n_spikes
            if n_spikes>n_spikes_max_exc:
                i_out_exc = i
                n_spikes_max_exc = n_spikes
        class_out_exc = np.floor(i_out_exc/(k_exc*N_RANKS))

        if class_out_exc==test_out[i_test]:
            count_unsupervised_right = count_unsupervised_right + 1
            
        print 'Corrects: ', count_unsupervised_right, '/', i_test+1
        print 'Accuracy = ', float(count_unsupervised_right)/float(i_test+1.)*100., '%'

        endTime = time.time()
        print ('Simulation time: %.2f s') % (endTime-startTime)
        sys.stdout.flush()
    
    Accuracy_pre_sleep = float(count_right)/float(i_test+1.)*100.
    Accuracy_pre_sleep_unsupervised = float(count_unsupervised_right)/float(i_test+1.)*100.
    
    accuracy_vector[n_cycle] = Accuracy_pre_sleep_unsupervised

    conn_par=nest.GetConnections(neur_exc, neur_exc)
    w_pre_sleep=nest.GetStatus(conn_par, ["source","target","weight"])
    
    conn_par_inp_exc=nest.GetConnections(neur_inp, neur_exc)
    w_inp_exc = nest.GetStatus(conn_par_inp_exc, ["source","target","weight"])

    sio.savemat('accuracy_vector_' +str(subset)+ '_thOn.mat', {'accuracy_vector':accuracy_vector})
    sio.savemat('w_' +str(subset)+ '_thOn' +str(n_cycle)+  '.mat', {'w':w_pre_sleep})
    sio.savemat('n_spikes_' +str(subset)+ '_thOn' +str(n_cycle)+  '.mat', {'n_spikes':fr_vector})

    print '######################################################################'
    print '# Sleep'
    print '######################################################################'
    print
    
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'alpha': alpha_sleep})
    nest.SetStatus(neur_exc, {"b": b_sleep})
    nest.SetStatus(neur_exc, {"tau_w": 400.0})
    nest.SetStatus(neur_inp, {"C_m": c_m_sleep})

    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'lambda': lambda_inp_exc_sleep})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'lambda': lambda_exc_out_sleep})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'lambda': lambda_exc_exc_sleep})
    nest.SetStatus(nest.GetConnections(source=neur_inh, target=neur_exc), {"weight": W_inh_exc_sleep})
    
    #switch off all teaching outputs
    for i in range(n_exc):
        nest.SetStatus([noise_exc[i]], {"rate": r_noise_exc_sleep})
    
    for i in range(n_inh):
        nest.SetStatus([noise_inh[i]], {"rate": 0.})
    
    for i in range(n_out):
        nest.SetStatus([noise_out[i]], {"rate": 0.0})
    
    #prepare input pattern
    for i in range(n_inp):
        nest.SetStatus([noise_inp[i]], {"rate": r_noise_inp*0.})
    
    #simulation
    nest.Simulate(t_sleep)
    
    conn_par=nest.GetConnections(neur_exc, neur_exc)
    w_post_sleep=nest.GetStatus(conn_par, ["source","target","weight"])

    evt_sd_inp = nest.GetStatus(sd_inp,keys="events")
    
    for i in range(n_inp):
        send_sd_inp = evt_sd_inp[i]["senders"]
        t_sd_inp = evt_sd_inp[i]["times"]
        n_spikes_inp[i] = len(t_sd_inp) - cum_spikes_inp[i];
        cum_spikes_inp[i] = len(t_sd_inp);
    evt_sd_out = nest.GetStatus(sd_out,keys="events")
    for i in range(n_out):
        send_sd_out = evt_sd_out[i]["senders"]
        t_sd_out = evt_sd_out[i]["times"]
        n_spikes_out[i] = len(t_sd_out) - cum_spikes_out[i];
        cum_spikes_out[i] = len(t_sd_out);
    evt_sd_exc = nest.GetStatus(sd_exc,keys="events")
    for i in range(n_exc):
        send_sd_exc = evt_sd_exc[i]["senders"]
        t_sd_exc = evt_sd_exc[i]["times"]
        n_spikes_exc[i] = len(t_sd_exc) - cum_spikes_exc[i];
        cum_spikes_exc[i] = len(t_sd_exc);
    evt_sd_inh = nest.GetStatus(sd_inh,keys="events")
    for i in range(n_inh):
        send_sd_inh = evt_sd_inh[i]["senders"]
        t_sd_inh = evt_sd_inh[i]["times"]
        n_spikes_inh[i] = len(t_sd_inh) - cum_spikes_inh[i];
        cum_spikes_inh[i] = len(t_sd_inh);
    evt_sd_inh2 = nest.GetStatus(sd_inh2,keys="events")
    for i in range(n_inh2):
        send_sd_inh2 = evt_sd_inh2[i]["senders"]
        t_sd_inh2 = evt_sd_inh2[i]["times"]
        n_spikes_inh2[i] = len(t_sd_inh2) - cum_spikes_inh2[i];
        cum_spikes_inh2[i] = len(t_sd_inh2);

    nest.SetStatus(neur_exc, {"b": 0.01})
    nest.SetStatus(neur_inp, {"C_m": 281.})
    
    nest.SetStatus(nest.GetConnections(source=neur_inp, target=neur_exc), {'lambda': 0.})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_out), {'lambda': 0.})
    nest.SetStatus(nest.GetConnections(source=neur_exc, target=neur_exc), {'lambda': 0.})
    
    nest.SetStatus(nest.GetConnections(source=neur_inh, target=neur_exc), {"weight": W_inh_exc_retrieval})

"""
######################################################################
#Plot

print '######################################################################'
print '# Plot'
print '######################################################################'
print

plt.figure()

ax1 = plt.subplot(2,1,1)
for i in range(n_out):
    plt.ylabel('# neuron out')

ax2 = plt.subplot(2,1,2)

for i in range(n_exc):
    plt.plot(evt_sd_exc[i]["times"],evt_sd_exc[i]["senders"], '.')
plt.ylabel('# neuron exc')

sys.stdout.flush()

plt.show()
"""
