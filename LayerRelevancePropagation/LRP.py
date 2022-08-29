from utils import *
from tqdm import tqdm
import numpy as np

def LRP_eps_nb(model, input, debug=False):
    # Function parameters initialization
    activations = get_activations(model, input)
    weights = model.coefs_
    relevance = [np.array(activations[-1]).flatten()]
    epsilon=1e-9
    # iterations
    for l in range(len(activations)-2, -1, -1): # l-2 corresponds to the (l-1)th layer
        # l : the l-th layer of current network
        current_activation = np.array(activations[l]).flatten() # (j,)
        current_weights = weights[l] # (j, k)
        prev_relevance = relevance[0] # each time update the 0-th term in relevance list, with shape (k, )
        if debug:
            print(len(prev_relevance))
        current_relevance = []
        for j in tqdm(range(len(current_activation))): # (j, )
            R_j = 0
            # Sum for calculating each Rj on this layer
            for k in range(len(current_weights.T)): # weights.T yield the shape of (k, j)
                # Get the numerator first
                if debug:
                    print(current_activation.shape, current_weights.shape, prev_relevance.shape)
                numerator = rou_function_nb(current_activation[j], current_weights[j][k], current_activation) * prev_relevance[k]
                denominator = epsilon
                for jj in range(len(current_activation)): # Loop through all the current neurons' contribution to give mean
                    if debug:
                        print(current_activation.shape)
                    #print("jj", jj)
                    #print("k", k)
                    denominator += rou_function_nb(current_activation[j], current_weights[jj][k], current_activation)
                R_j += (numerator / denominator) # Sum up k elements for R_j
            current_relevance.append(R_j) # Update current relevance with R_js
        relevance.insert(0, current_relevance) # Update current relevance to top of relevance list of different layers
    return relevance