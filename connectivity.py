'''
    library for building network structure
    includes different connectivity patterns and algorithms
    should be used together with Brian simulator
    Started October 2013 by Hesam SETAREH
    last modification: Sep 4, 2014
    last modification : April 12, 2018 by Sharbat
'''
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import brian2
import brian2.synapses
import time

# IMPLEMENT WIRING ALGORITHM
# TEST WIRING ALGORITHM
# TEST VARIATION MATRIX


def sort_array_column_based(array):
    '''
        Sort columns of an array based on sum of each column
    '''
    N = numpy.shape(array)[1]
    
    for i in range(N):
        for j in range(i,N):
            if numpy.sum(array[:,i]) > numpy.sum(array[:,j]):
                swap_columns(array,i, j)
    

def swap_columns(array, c_index1, c_index2):
    '''
        Swaping two columns of an array
    '''
    temp = numpy.array(array[:,c_index1])
    array[:,c_index1] = array[:,c_index2]
    array[:,c_index2] = temp

def random_generator(prob):
    '''
       Choosing a number from [0,len(prob)) based on the given probabilities
       Number i will be selected with the probability prob[i]
    '''
    prob_acc = numpy.cumsum(prob)
    x = numpy.random.random_sample(1)
    indices = numpy.where(x < prob_acc)
    return numpy.min(indices)

def connection_matrix_variation_new(M, N, p, pre_lambda=0, post_lambda=0,self_conn=False):
    '''
        For constructing binary(0-1) connectivity matrix with constant probability p
        Produces a network with degree variation
        Each of pre/post could have either power-law or bionomial distribution
        std=0 ==> bionomial distribution, std!=0 ==> power-law distribution
        
        M: presynaptic population size
        N: postsynaptic population size
    '''
    
    if pre_lambda!=0:
        tmp = numpy.arange(1,M+1)
        pre_prob = (1.0*pre_lambda/M)*numpy.exp(-tmp*1.0*pre_lambda/M)
        pre_prob /= numpy.sum(pre_prob) # scaling
    else:
        pre_prob = numpy.ones(M)/M

    if post_lambda!=0:
        tmp = numpy.arange(1,N+1)
        post_prob = (1.0*post_lambda/N)*numpy.exp(-tmp*1.0*post_lambda/N)
        post_prob /= numpy.sum(post_prob) # scaling
    else:
        post_prob = numpy.ones(N)/N
    
    
    edges_number = int(M*N*p)
    
    pre_indices = numpy.zeros(edges_number, dtype=int)
    post_indices = numpy.zeros(edges_number, dtype=int)

    matrix = numpy.zeros([M,N], dtype=int)
    
    i = 0
    
    while i<edges_number:
        pre_indices[i] = random_generator(pre_prob)
        post_indices[i] = random_generator(post_prob)
        if (self_conn==False and pre_indices[i]==post_indices[i]) or matrix[pre_indices[i],post_indices[i]]==1:
            continue
        
        matrix[pre_indices[i]][post_indices[i]]=1
        i+=1
        #print i
        
             
    #return pre_indices, post_indices, matrix
    return matrix


def connection_matrix_variation(M, N, p, pre_lambda=0, post_lambda=0):
    '''
        For constructing binary(0-1) connectivity matrix with constant probability p
        Produces a network with degree variation
        Each of pre/post could have either power-law or bionomial distribution
        std=0 ==> bionomial distribution, std!=0 ==> power-law distribution
        
        M: presynaptic population size
        N: postsynaptic population size
    '''
    
    pre_mean = p*(N-1)
    post_mean = p*(M-1)
    
    if pre_lambda!=0:
        u = numpy.random.power(pre_lambda,M) * (N-1) # u is the degree array of presynaptic neurons
        u = u.astype(int)
    else:
        u = numpy.random.binomial(N-1, p, M)

    if pre_lambda!=0:
        v = numpy.random.power(post_lambda,N) * (M-1) # v is the degree array of postsynaptic neurons
        v = v.astype(int)
    else:
        v = numpy.random.binomial(M-1, p, N)
        
    U = numpy.sum(u)
    V = numpy.sum(v)
    
    if U!=V:
        addremove(u, v)
    
    print (numpy.sum(u)==numpy.sum(v))  # should be removed
        
    matrix = wiring(u, v)    # should be revised
    
    return matrix

def addremove(u, v):
    
    U = numpy.sum(u)
    V = numpy.sum(v)
    
    while U!=V:
        rand1 = numpy.random.binomial(1, 0.5, 1)
        if rand1==0:
            sign = -numpy.sign(U-V)
            select_randomly(u,sign)
            U = U + 1*sign
        else:
            sign = -numpy.sign(V-U)
            select_randomly(v,sign)
            V = V + 1*sign

def wiring(pre_degree, post_degree):   # should be implemented
    M = len(pre_degree)
    N = len(post_degree)
    
    
    
def select_randomly(array, sign):
    sum = numpy.sum(array)
    rand = numpy.random.uniform(0, sum)
    cum_array = numpy.cumsum(array)
    
    index = len(array)-1
    
    for i in range(0,index+1):
        if cum_array[i]>rand:
            index = i
            break
        
    array[index] = array[index] + 1*sign        
        
def connection_matrix_random(M, N, p, self_conn=False):
    '''
        For constructing binary(0-1) connectivity matrix with constant probability p
        Produces a random network (Erdos-Renyi)
        M: presynaptic population size
        N: postsynaptic population size
    '''
    matrix = numpy.random.binomial(1, p, [M, N])

    if self_conn==False:
        for i in range(0,min(M,N)):
            matrix[i,i] = 0
            
    return matrix

def normal_weight(M, N, mean, std, self_conn=False):
    '''
        For constructing weight matrix using nomral distribution
        need be multiplied by connection matrix
        M: presynaptic population size
        N: postsynaptic population size
    '''
    matrix = numpy.random.normal(mean, std, [M,N])
    
    if self_conn==False:
        for i in range(0,min(M,N)):
            matrix[i,i] = 0
            
    return matrix
            
def lognormal_weight(M, N, mean, std, self_conn=False):
    '''
        For constructing weight matrix using lognomral distribution
        need be multiplied by connection matrix
        M: presynaptic population size
        N: postsynaptic population size
    '''
    matrix = numpy.random.lognormal(mean, std, [M,N])
    
    if self_conn==False:
        for i in range(0,min(M,N)):
            matrix[i,i] = 0
            
    return matrix

def weight_corr_inward2(matrix1, matrix2, mean, std, sort=False):
    '''
        matrix1 : L2/3 --> L2/3   N*N
        matrix2 : L5A  --> L2/3   M*N
        For adding inward correlation on two weight matrices
    '''
    M = matrix2.shape[0]
    N = matrix2.shape[1]
    
    coeff = numpy.random.lognormal(mean, std, N)
    
    if sort==True:
        coeff.sort()
    
    for i in range(0,N):
        matrix1[:,i] = matrix1[:,i]*coeff[i]
        matrix2[:,i] = matrix2[:,i]*coeff[i]
        
    return matrix1, matrix2


def weight_corr_inward(matrix, mean, std, sort=False):
    '''
        For adding inward correlation on weight matrix
    '''
    M = matrix.shape[0]
    N = matrix.shape[1]
    
    coeff = numpy.random.lognormal(mean, std, N)
    
    if sort==True:
        coeff.sort()
    
    for i in range(0,N):
        matrix[:,i] = matrix[:,i]*coeff[i]
        
    return matrix

def weight_corr_outward(matrix, mean, std, sort=False):
    '''
        For adding outward correlation on weight matrix
    '''
    M = matrix.shape[0]
    N = matrix.shape[1]
    
    coeff = numpy.random.lognormal(mean, std, M)

    if sort==True:
        coeff = numpy.sort(coeff)
    
    for i in range(0,M):
        matrix[i] = matrix[i]*coeff[i]
        
    return matrix
    
def lognfit(x):
    '''
        For estimating lognormal_mean and lognormal_std of a dataset
    '''
    mean_real = numpy.mean(x)
    std_real = numpy.std(x)
    var_real = std_real**2
 
    var_log_hat = numpy.log(var_real/(mean_real**2)+1)
    mean_log_hat = numpy.log(mean_real)-var_log_hat/2
    std_log_hat = numpy.sqrt(var_log_hat)

    return mean_log_hat, std_log_hat

def hub_sorting(matrix): #does not yield demanding result!
    '''
    implements Moritz idea for making hubs
    '''
    M = matrix.shape[0]
    N = matrix.shape[1]
    
    for i in range(0,M):
        matrix[i] = numpy.sort(matrix[i])
        
    for i in range(0,N):
        matrix[:,i] = numpy.sort(matrix[:,i])
        
    return matrix
        

def find_hubs(matrix, threshold, hub='inward'):
    '''
    find index number of neurons which receives (inward) or 
    sends (outward) weight connections more than the threshold
    '''
    
    if hub=='inward':
        array = numpy.sum(matrix,0)
    elif hub=='outward':
        array = numpy.sum(matrix,1)
    else:
        raise Exception('hub should be inward or outward!')
    
    indices = numpy.nonzero(array>threshold)
    return indices[0]
    
def subnetwork_analysis(matrix, subnet):
    '''
    find connection probabilities and mean weight
    inside a subnetwork (usually hubs)
    matrix should weight*connectivity
    '''
    subnet_len = len(subnet)
    connection_counter = 0
    weight_sum = 0
    
    for i in range(0,subnet_len):
        for j in range(0,subnet_len):
            if matrix[subnet[i]][subnet[j]]!=0:
                connection_counter = connection_counter + 1
                weight_sum = weight_sum + matrix[subnet[i]][subnet[j]]
    
    return (1.0*connection_counter)/subnet_len**2, weight_sum/connection_counter

def choose_neuron(size_pop, size_sel):
    '''
    choosing size_sel number from interval [0, size_pop] randomly
    '''
    
    numpy.random.seed(4)
    
    result = numpy.zeros(size_sel, dtype = int)
    result[0] = numpy.int(numpy.random.uniform(0, size_pop))
    
    i=0
    while i<size_sel:
        x = numpy.int(numpy.random.uniform(0, size_pop))
        if result[:i].__contains__(x)==False:
            result[i] = x
            i += 1
        else:
            continue
    
    return result
  
def dense_hub(matrix, hubs, edge):
    '''
    remove edge between non-hubs and add it to hubs
    matrix: binary connectivity matrix
    hubs: index of hubs
    edge: number of edge
    '''
    
    pop_size = matrix.shape[0]
    edge_counter = 0
    hubs_size = len(hubs)
    
    while edge_counter < edge:
        pre_x  = numpy.int(numpy.random.uniform(hubs_size))
        post_x = numpy.int(numpy.random.uniform(hubs_size))
        
        if post_x==pre_x or matrix[hubs[pre_x],hubs[post_x]]==1:
            continue
        else:
            matrix[hubs[pre_x],hubs[post_x]]=1
            
        edge_counter += 1
    '''    
    while edge_counter > 0:
        pre_x  = numpy.int(numpy.random.uniform(pop_size))
        if hubs.__contains__(pre_x):
            continue
        else:
            temp =  matrix[pre_x].nonzero()[0]
            post_x = temp[edge_counter%temp.__len__()]
            matrix[pre_x][post_x] = 0
        
        edge_counter -= 1
    '''    
        
        
        
        
    
def plot_degree_dist(matrix, degree='in'):
    '''20
    plot the distribution of in-degree or out-degree
    matrix: binary connectivity matrix
    '''
    
    if degree=='in':
        array = numpy.sum(matrix,0)
    elif degree=='out':
        array = numpy.sum(matrix,1)
    else:
        raise Exception('degree should be in or out!')
    
    plot_hist(array,100)
    
 
def plot_matrix(matrix):
    '''
    shows a matrix (for instance weight matrix) in a figure
    '''
    fig, ax = plt.subplots()
    M = matrix.shape[0]
    N = matrix.shape[1]
    
    plt.register_cmap(cmap=plt.cm.gray)
    
    a = ax.imshow(matrix, interpolation='nearest')
    plt.colorbar(a)

    ax.set_title('Matrix')
    
    plt.ylabel("presynaptic")
    plt.xlabel("postsynaptic")
    
    
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    plt.show()

def normal_to_lognormal(normal_mean, normal_std, plot=False):
    '''
    return lognormal_mean and lognormal_std produced by that lognormal
    using real mean and real std
    plot histogram if plot==True
    '''
    normal_var = normal_std**2
    lognormal_var = numpy.log(normal_var/normal_mean**2 + 1)
    lognormal_mean = numpy.log(normal_mean) - lognormal_var/2
    lognormal_std = numpy.sqrt(lognormal_var)
    median = numpy.exp(lognormal_mean)  #  median is same for both normal and lognormal
    
    if plot:
        data = numpy.random.lognormal(lognormal_mean,lognormal_std,1000000)
        plot_hist(data,80)
        print (numpy.mean(data))
        
    return lognormal_mean, lognormal_std, median


def lognormal_to_normal(lognormal_mean, lognormal_std):
    '''
    return normal_mean and normal_std from normal_mean and normal_std
    '''
    normal_mean = numpy.exp(lognormal_mean + lognormal_std**2/2)
    normal_std = (numpy.exp(lognormal_std**2)-1) * numpy.exp(2*lognormal_mean + lognormal_std**2)
    
    return normal_mean, normal_std


def plot_hist(data, bin):    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    n, bins = numpy.histogram(data, bin,[0,8]) # range changed

# get the corners of the rectangles for the histogram
    left = numpy.array(bins[:-1])
    right = numpy.array(bins[1:])
    bottom = numpy.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = numpy.array([[left,left,right,right], [bottom,top,top,bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='white', edgecolor='gray', alpha=0.9)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.show()
    

def to_new_lognormal(data):
    w_m = numpy.median(data) 
    data /= w_m
    miu, sigma = lognfit(data)
    
    return w_m, miu,sigma
     

#data = numpy.loadtxt("l5a_exc_exc_weights_lefort.txt")*1000
#data = numpy.random.lognormal(-0.8651, 0.9386,100000)
#print to_new_lognormal(data)





