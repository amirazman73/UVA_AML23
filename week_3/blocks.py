import numpy as np

#######################################################
# put `w2_sigmoid_forward` and `w2_sigmoid_grad_input` here #
#######################################################

def w2_sigmoid_forward(x_input):
    return 1 / (1 + np.exp(-x_input))

def w2_sigmoid_grad_input(x_input, grad_output):
    return grad_output * (1-grad_output)

#######################################################
# put `w2_nll_forward` and `w2_nll_grad_input` here    #
#######################################################
