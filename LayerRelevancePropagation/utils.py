import numba as nb

def model_summary(model, trained=False):
    # Gradient
    weight_shape = [weight.shape for weight in model.coefs_]
    print("Number of Coefs (layer): ", len(model.coefs_))
    print(weight_shape)
    print("="*25)
    # Bias
    intercept_shape = [intercept.shape for intercept in model.intercepts_]
    print("Number of Intercepts (layer): ", len(model.intercepts_))
    print(intercept_shape)
    print("="*25)
    # Hidden layer activation
    print("Hidden layer activation function : ", model.activation)
    # Output layer activation
    print("Output layer activation function : ", model.out_activation_)
    print("="*25)
    if trained:
        print("Training information: ")
    # For trined models
        # loss
        print("Loss : ", model.loss_)
        # iter time
        print("Number of Iterations for Which Estimator Ran : ", model.n_iter_)

def get_activations(model, inputs):
    # Utils
    from sklearn.utils.extmath import safe_sparse_dot
    from sklearn.neural_network._base import (
        inplace_identity as identity,
        inplace_tanh as tanh,
        inplace_logistic as logistic,
        inplace_softmax as softmax,
        inplace_relu as relu
    )
    ACTIVATIONS = {
    "identity": identity,
    "tanh": tanh,
    "logistic": logistic,
    "relu": relu,
    "softmax": softmax,
    }
    # Input Validation
    print("Warning: The input model must be trained before doing so, or the result is not meaningful")
    assert(len(model.coefs_[0]) == len(inputs.T)), "Input layer shape {} and feature shape {} not match"\
        .format(len(model.coefs_[0]), len(inputs.T))
    activations = []
    activations.append(inputs) # Add the input level as the layer zero activations
    # Initialize first layer
    activation = inputs

    # Forward propagate
    hidden_activation = ACTIVATIONS[model.activation]
    for i in range(model.n_layers_ - 1):
        activation = safe_sparse_dot(activation, model.coefs_[i])
        activation += model.intercepts_[i]
        if i != model.n_layers_ - 2:
            hidden_activation(activation)
            # Store the hidden layer activation result
            activations.append(activation)
            print("Append hidden")
    output_activation = ACTIVATIONS[model.out_activation_]
    output_activation(activation)
    # Store the output layer activation result
    activations.append(activation)
    print("Appedn output")
    
    return activations

@nb.njit()
def rou_function_nb(a, w, a_list):
    """
    a: value
    w: value
    a_list: list of activations in the same layer
    """
    # get the lower and upper activations
    l = min(a_list)
    h = max(a_list)
    # get the likely-relu activated value of weights #TODO: Work out why it looks like this
    w_l = max([0, w]) # RELU
    w_h = min([0, w]) # Negative RELU
    return (a*w - l*w_l - h*w_h)