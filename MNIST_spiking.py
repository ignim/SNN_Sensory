import numpy as np
#import cupy as cp
import matplotlib.cm as cmap
import time
import math
import os.path
from numpy.random.mtrand import randint
import scipy
from scipy.stats import mode
import pickle as pickle
from struct import unpack
import matplotlib.pyplot as plt
import matplotlib.image as img

MNIST_data_path = 'mnist/'
weight_data_path = 'random/'

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels-idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images-idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint16)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint16)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def poisson_spike_train(rate, interval):
    lam = rate * time_step / 8. * interval
    P =  np.random.uniform(0,1,(int(n_input), int(spike_rate_per_time))).astype(np.float32)
    spike = np.where(P < lam[:,None], 1,0).astype(np.float32)
    return spike

def input_spike(weight, spike):
    output = np.matmul(weight, spike.reshape(int(n_input),1,int(spike_rate_per_time))).astype(np.float32)
    return output


def E_spike_gen(Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE, Sensory_ge_spike,weight_excitation_inhibition, weight_inhibition_excitation, Sensory_gE_max,times, theta):
    Sensory_spike = np.zeros((n_e, int(spike_rate_per_time)), dtype=np.uint16)
    Inter_I_spike = np.zeros((n_e, int(spike_rate_per_time)+1), dtype=np.uint16)
    for i in range(int(times)):
        v_thresh_e = -0.055 * np.ones(n_e, dtype=np.float32) #[V]
        if i < 5:
            v_thresh_e = v_thresh_e + theta - 0.02
            spike_part_sum = np.sum(Sensory_spike[:,0:i], axis = 1).astype(np.uint16)
            S_not_neuron = np.where(spike_part_sum != 0)
            S_neuron = np.where(spike_part_sum == 0)
            I_to_E_spike_data = np.sum(weight_inhibition_excitation * Inter_I_spike[:,i], axis= 1)
            Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max
            Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max
            Sensory_dv = (-(Excitation_potential - v_rest_e) - (Sensory_gE/gL)*(Excitation_potential-vE_E)- (Sensory_gI/gL) * (Excitation_potential - vI_E))*(time_step /tau_e)
            Excitation_potential[S_neuron] = Excitation_potential[S_neuron] + Sensory_dv[S_neuron]
            Excitation_potential[S_not_neuron] = v_rest_e
            Sensory_spike[:,i] =  np.where(v_thresh_e < Excitation_potential,1,0).astype(np.float32)
            Excitation_potential =  np.where(v_thresh_e < Excitation_potential, v_reset_e, Excitation_potential).astype(np.float32)
            Sensory_spike[S_not_neuron,i] = 0
            if th ==1 and adp == 1:
                theta = np.where(Sensory_spike[:,i] == 1, theta + 0.00005, theta).astype(np.float32)
                theta_dv = -theta/(10**6.1)
                theta = theta + theta_dv
            elif th ==0 and adp == 1:
                Sensory_gE_max = np.where(Sensory_spike[:,i] == 1, Sensory_gE_max * 0.9991, Sensory_gE_max).astype(np.float32)
                Sensory_gE_max_dv = Sensory_gE_max/(10**8.1)
                Sensory_gE_max = Sensory_gE_max + Sensory_gE_max_dv
        elif i >= 5:
            v_thresh_e = v_thresh_e + theta - 0.02
            I_to_E_spike_data = np.sum(weight_inhibition_excitation * Inter_I_spike[:,i], axis= 1)
            spike_part_sum = np.sum(Sensory_spike[:,i-5:i], axis = 1)
            S_not_neuron = np.where(spike_part_sum != 0)
            S_neuron = np.where(spike_part_sum == 0)
            Sensory_gE = Sensory_gE * (1 - time_step / tau_syn_E) + np.sum(Sensory_ge_spike[:,:,i], axis = 0) * Sensory_gE_max
            Sensory_gI = Sensory_gI * (1 - time_step / tau_syn_I) + I_to_E_spike_data * Sensory_gI_max
            Sensory_dv = (-(Excitation_potential - v_rest_e) - (Sensory_gE/gL)*(Excitation_potential-vE_E)- (Sensory_gI/gL) * (Excitation_potential - vI_E))*(time_step /tau_e)
            Excitation_potential[S_neuron] = Excitation_potential[S_neuron] + Sensory_dv[S_neuron]
            Excitation_potential[S_not_neuron] = v_rest_e
            Sensory_spike[:,i] =  np.where(v_thresh_e < Excitation_potential,1,0).astype(np.float32)
            Excitation_potential =  np.where(v_thresh_e < Excitation_potential, v_reset_e, Excitation_potential).astype(np.float32)
            Sensory_spike[S_not_neuron,i] = 0
            if th ==1 and adp == 1:
                theta = np.where(Sensory_spike[:,i] == 1, theta + 0.00005, theta).astype(np.float32)
                theta_dv = -theta/(10**6.1)
                theta = theta + theta_dv 
            elif th ==0 and adp == 1:
                Sensory_gE_max = np.where(Sensory_spike[:,i] == 1, Sensory_gE_max * 0.9991, Sensory_gE_max).astype(np.float32) #-0.0007 0.995 0.9987 0.9999
                Sensory_gE_max_dv = Sensory_gE_max/(10**8.1) #6.07, 6.23 6 7.5
                Sensory_gE_max = Sensory_gE_max + Sensory_gE_max_dv
        if i < 2:
            E_to_I_spike_data = weight_excitation_inhibition * Sensory_spike[:,i]
            spike_part_sum = np.sum(Inter_I_spike[:,0:i], axis = 1).astype(np.uint16)
            I_not_neuron = np.where(spike_part_sum != 0)
            I_neuron = np.where(spike_part_sum == 0)
            Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max
            Inter_dv_I = (-(inhibition_potential - v_rest_i) - (Inter_I_gE/gL)*(inhibition_potential-vE_I))*(time_step /tau_e)
            inhibition_potential[I_neuron] = inhibition_potential[I_neuron] + Inter_dv_I[I_neuron]
            inhibition_potential[I_not_neuron] = v_rest_i
            Inter_I_spike[:,i+1] =  np.where(v_thresh_i < inhibition_potential,1,0).astype(np.float32)
            inhibition_potential =  np.where(v_thresh_i < inhibition_potential, v_reset_i, inhibition_potential).astype(np.float32)
            Inter_I_spike[I_not_neuron,i+1] = 0
        elif i >= 2:
            E_to_I_spike_data = weight_excitation_inhibition * Sensory_spike[:,i]
            spike_part_sum = np.sum(Inter_I_spike[:,i-2:i], axis = 1)
            I_not_neuron = np.where(spike_part_sum != 0)
            I_neuron = np.where(spike_part_sum == 0)
            Inter_I_gE = Inter_I_gE * (1 - time_step / tau_syn_E) + E_to_I_spike_data * Inter_I_gE_max
            Inter_dv_I = (-(inhibition_potential - v_rest_i) - (Inter_I_gE/gL)*(inhibition_potential-vE_I))*(time_step /tau_e)
            inhibition_potential[I_neuron] = inhibition_potential[I_neuron] + Inter_dv_I[I_neuron]
            inhibition_potential[I_not_neuron] = v_rest_i
            Inter_I_spike[:,i+1] =  np.where(v_thresh_i < inhibition_potential,1,0).astype(np.float32)
            inhibition_potential =  np.where(v_thresh_i < inhibition_potential, v_reset_i, inhibition_potential).astype(np.float32)
            Inter_I_spike[I_not_neuron,i+1] = 0
    return Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE,Sensory_spike, Sensory_gE_max, theta ,Inter_I_spike

def find_nearest_over(array, value):
    over_array = array[np.where(array >= value)]
    if over_array.shape == (0,):
        over_value = None
    else :
        over_value = over_array[(np.abs(over_array - value)).argmin()]
    return over_value

def find_nearest_under(array, value, nueron_train_count):
    under_array = array[np.where(array <= value)]
    if under_array.shape == (0,):
        under_value = None
    else :
        under_value = under_array[(np.abs(under_array - value)).argmin()]
        if nueron_train_count[np.where(array == under_value)] == 0:
            nueron_train_count[np.where(array == under_value)] = 1
        elif nueron_train_count[np.where(array == under_value)] == 1:
            under_value = None
    return under_value

def STDP(pre, post, winner_num, Weight):
    a,b,c = np.where(pre == 1)
    d,e = np.where(post == 1)
    pre_time = np.stack((a,b,c)).T
    post_time = np.stack((d,e)).T
    for i in np.nditer(winner_num):
        pre_time_arr = pre_time[np.where(pre_time[:,1]==i)]
        post_time_arr = post_time[np.where(post_time[:,0]==i)]
        pre_time_arr_input = np.unique(pre_time_arr[:,0])
        if post_time_arr.shape != (0,):
            for j in np.nditer(pre_time_arr_input):
                pre_time_arr_temp_in = pre_time_arr[np.where(pre_time_arr[:,0] == j)]
                setting = np.zeros(len(pre_time_arr_temp_in))
                for k in np.nditer(post_time_arr[:,1]):
                    under = find_nearest_under(pre_time_arr_temp_in[:,2],k,setting)
                    if k == under:
                        w_del = 0
                    elif under is None:
                        over = find_nearest_over(pre_time_arr_temp_in[:,2],k)
                        if k == over:
                            w_del = 0
                        elif over is None:
                            w_del = -np.exp(-1/5.)                
                        else :
                            w_del = -np.exp((k-over)/40.) # x<0
                    else :
                        w_del = np.exp(-(k-under) / 20.) # x >0
                    if w_del == 0:
                        Weight[j,i] = Weight[j,i]
                    elif w_del < 0:
                        if Weight[j,i] < np.abs(0.008*w_del*(Weight[j,i]**0.9)):
                            Weight[j,i] = 0
                        else:
                            Weight[j,i] = Weight[j,i] + 0.008*w_del*(Weight[j,i]**0.9)
                    else :
                        Weight[j,i] = Weight[j,i] + 0.008*w_del*((1-Weight[j,i])**0.9)
    return Weight

def spike_count(spike):
    count = np.sum(spike, axis= 1)
    return count

def expect_number(count, nue_expect):
    expect = nue_expect[np.where(count == np.max(count))]
    expect_num = mode(expect)[0][0]
    return count, expect_num

def winner(count):
    count_copy = np.copy(count)
    winner_num = np.array(np.where(count_copy == np.max(count_copy))).reshape(-1).astype(np.uint16)
    return winner_num

def normalization(weight):
    weight_temp = np.copy(weight)
    Colsums = np.sum(weight_temp, axis= 0)
    Colfactors = 85.5 / Colsums
    weight = np.multiply(weight_temp, Colfactors)
    return weight
#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------

training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)

training_data = np.array(training['x']).astype(np.float32)
training_label = np.array(training['y']).astype(np.uint16)
testing_data = np.array(testing['x']).astype(np.float32)
testing_label = np.array(testing['y']).astype(np.uint16)

n_input = 784
sim_time = 0.350
time_step = 0.001
spike_rate_per_time = 350
n_e = 625
n_i = n_e


performance_count = 0.

v_rest_e = -0.065   #[V]
v_rest_i = -0.06    #[V]
v_reset_e = -0.080  #[V]
v_reset_i = -0.075  #[V]

v_thresh_i = -0.055  #[V]
refrac_e = 0.005    #[s]
refrac_i = 0.002    #[s]

tau_e = 0.1         #[s]
tau_i = 0.01        #[s]

tau_syn_E = 0.001   #[s]
tau_syn_I = 0.002    #[s]

Sensory_gI_max = np.ones(n_e,dtype = np.float32)  #[nS]

gL = 1.      #[nS]       
vE_E = 0.

vE_I = 0.

Excitation_potential = np.ones(n_e, dtype=np.float32) * -0.065      #[V]
inhibition_potential = np.ones(n_e, dtype=np.float32) * -0.06  #[V]


initial_Excitation_potential = np.copy(Excitation_potential).astype(np.float32)
initial_inhibition_potential = np.copy(inhibition_potential).astype(np.float32)

Sensory_gE = np.zeros(n_e, dtype=np.float32)
Sensory_gI = np.zeros(n_e, dtype=np.float32)
Inter_I_gE = np.zeros(n_e, dtype=np.float32)

relax = np.zeros((784,n_e,350))

weight_input_excitation = np.copy(np.load(weight_data_path + 'X_to_Sen.npy')).astype(np.float32)
weight_excitation_inhibition = np.copy(np.load(weight_data_path + 'Sen_in_E.npy')).astype(np.float32)
weight_inhibition_excitation = np.copy(np.load(weight_data_path + 'I_to_X.npy')).astype(np.float32)

training_iter = 180000
winner_num = 0
interval = 2.
a = np.zeros(10,dtype = np.uint16)
train = 1
th = 0
start = time.time()
if train == 1:
    total_count = 0
    total_acc = 0
    adp = 1
    theta = 0.02 * np.ones(n_e,dtype = np.uint16)#[V]
    Sensory_gE_max = np.ones(n_e,dtype = np.uint16)  #[nS]
    Inter_I_gE_max = np.ones(n_e,dtype = np.uint16) * 300 #[nS]
    vI_E = -0.250
    neuron_expect = np.zeros(n_e,dtype = np.uint16)
    neuron_fire_num = np.zeros((n_e,10),dtype = np.uint16)
    if th == 1:
        print('start train, th')
    if th == 0:
        print('start train, Sen')
    for i in range(training_iter):
        real_num = training_label[i%60000][0]
        data = training_data[i%60000,:,:].reshape(n_input)
        count = np.zeros(n_e, dtype=np.float32)
        start_Excitation_potential = np.copy(Excitation_potential).astype(np.float32)
        start_inhibition_potential = np.copy(inhibition_potential).astype(np.float32)
        start_theta = np.copy(theta).astype(np.float32)
        weight_input_excitation = normalization(weight_input_excitation)
        if i == 0 or i % 10000 == 9999:
            fig, axes = plt.subplots(nrows= 25,ncols= 25, figsize = (25,25))
            weight_show = np.reshape(weight_input_excitation, (784,n_e))
            for j, ax in zip(range(n_e),axes.flat):
                weight_show_1 = weight_show[:,j].reshape(28,28)
                im = ax.imshow(weight_show_1, vmin = 0, vmax = 1, cmap = cmap.get_cmap('hot_r'))
                ax.set_yticks([])
                ax.set_xticks([])
            fig.subplots_adjust(right = 0.835)
            cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
            fig.colorbar(im, cax = cbar_ax)
            if th == 1:
                plt.savefig('weight_th/' + str(i+1) + '.jpg',dpi = 300)
            else:
                plt.savefig('weight_Sen/' +str(i+1) + '.jpg',dpi = 300, bbox_inches = 'tight')
        while np.sum(count) < 5 :
            if i == 0:
                Excitation_potential = initial_Excitation_potential
                inhibition_potential = initial_inhibition_potential
                theta = start_theta
            else:
                Excitation_potential = start_Excitation_potential
                inhibition_potential = start_inhibition_potential
                theta = start_theta
            current_spike = poisson_spike_train(data, interval)
            Sensory_ge_spike = input_spike(weight_input_excitation, current_spike)
            Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE,Sensory_spike, Sensory_gE_max, theta, Inter_I_spikes = E_spike_gen\
                (Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE, Sensory_ge_spike,weight_excitation_inhibition, weight_inhibition_excitation, Sensory_gE_max,300, theta)
            count = spike_count(Sensory_spike)
            if np.sum(count) < 5:
                interval += 1.
        count, num_expect = expect_number(count, neuron_expect)
        winner_num = winner(count)
        neuron_fire_num[winner_num,real_num] += 1
        a[real_num] += 1
        if np.sum(count) != 0:
            interval = 2.
        pre_spike = np.where(Sensory_ge_spike != 0 , 1, 0)
        weight_input_excitation = STDP(pre_spike, Sensory_spike, winner_num, weight_input_excitation)
        Excitation_potential, inhibition_potential,relax_Sensory_gE, relax_Sensory_gI, Inter_I_gE,relax_Sensory_spike, relax_Sensory_gE_max, relax_theta, relax_Inter_I_spike = E_spike_gen\
                (Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE, \
                    relax,weight_excitation_inhibition, weight_inhibition_excitation, Sensory_gE_max,50, theta)
        total_count += np.sum(count)+np.sum(Inter_I_spikes)
        if num_expect == real_num:
            performance_count += 1
        if i % 10000 == 9999:
            neuron_expect = np.argmax(neuron_fire_num/a, axis = 1)
            unique, counts = np.unique(neuron_expect, return_counts=True)
            acc = (performance_count / 10000)*100
            print('iter : ',(i+1),'accuracy = ',acc,'%')
            performance_count = 0
            neuron_fire_num = np.zeros((n_e,10),dtype = np.uint16)
            a = np.zeros(10,dtype = np.uint16)
            #if th == 0:
            #    print(np.mean(Sensory_gE_max))
            #else :
            #    print(np.mean(v_thresh_e))
            #total_count = 0
            print('train',time.time() - start)
    if th == 1:
        np.save('weight_th/weight',weight_input_excitation)
        np.save('weight_th/neuron_expect',neuron_expect)
        np.save('weight_th/neuron_fire_num',neuron_fire_num)
    #np.save('weight_Sen/Sensory_gE_max', Sensory_gE_max)
    else:
        np.save('weight_Sen/weight',weight_input_excitation)
        np.save('weight_Sen/neuron_expect',neuron_expect)
        np.save('weight_Sen/neuron_fire_num',neuron_fire_num)
        np.save('weight_Sen/Sensory_gE_max', Sensory_gE_max)
    print('train',time.time() - start)
else:
    adp = 0
    total_count = 0
    Inter_I_gE_max = np.ones(n_e,dtype = np.uint16) * 300 #[nS]
    vI_E = -0.25
    if th == 1:
        print('start test, th, 0.069')
    if th == 0:
        print('start test, Sen, 0.001')
    if th == 1:
        weight_input_excitation = np.copy(np.load('weight_th/' + 'weight.npy'))
        neuron_expect = np.copy(np.load('weight_th/' + 'neuron_expect.npy'))
        neuron_fire_num = np.copy(np.load('weight_th/' + 'neuron_fire_num.npy'))
        theta = 0.069 * np.ones(n_e,dtype = np.uint16) #np.copy(np.load('weight_th/' + 'theta_th.npy'))
        Sensory_gE_max = np.ones(n_e,dtype = np.uint16)
    else:
        weight_input_excitation = np.copy(np.load('weight_th/' + 'weight.npy')) #weight_Sen/
        neuron_expect = np.copy(np.load('weight_th/' + 'neuron_expect.npy'))
        neuron_fire_num = np.copy(np.load('weight_th/' + 'neuron_fire_num.npy'))
        theta = 0.02
        Sensory_gE_max = 0.001#np.copy(np.load('weight_Sen/' + 'Sensory_gE_max.npy'))
    performance_count = 0
    start = time.time()
    for i in range(10000):
        start_Excitation_potential = np.copy(Excitation_potential)
        start_inhibition_potential = np.copy(inhibition_potential)
        start_theta = np.copy(theta)
        count = np.zeros(n_e)
        data = testing_data[i%10000,:,:].reshape(n_input)
        real_num = testing_label[i%10000,:]
        while np.sum(count) < 5 :
            if i == 0:
                Excitation_potential = initial_Excitation_potential
                inhibition_potential = initial_inhibition_potential
                theta = start_theta
            else:
                Excitation_potential = start_Excitation_potential
                inhibition_potential = start_inhibition_potential
                theta = start_theta
            current_spike = poisson_spike_train(data, interval)
            Sensory_ge_spike = input_spike(weight_input_excitation, current_spike)
            Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE,Sensory_spike, Sensory_gE_max, theta, Inter_I_spikes = E_spike_gen\
                (Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE, Sensory_ge_spike,weight_excitation_inhibition, weight_inhibition_excitation, Sensory_gE_max,350, theta)
            count = spike_count(Sensory_spike)
            if np.sum(count) < 5:
                interval += 1.
        count, num_expect = expect_number(count, neuron_expect)
        winner_num = winner(count)
        Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE,Sensory_spike, Sensory_gE_max, theta, Inter_I_spikes = E_spike_gen\
                (Excitation_potential, inhibition_potential,Sensory_gE, Sensory_gI, Inter_I_gE, Sensory_ge_spike,weight_excitation_inhibition, weight_inhibition_excitation, Sensory_gE_max,300, theta)        
        if np.sum(count) > 5:
            interval = 2.
        if num_expect == real_num:
            performance_count += 1
    print('accuracy = ',(performance_count / 10000)*100,'%')
    print('end',time.time() - start)
