
def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]    # number of channels
    # return ((1-epsilon) * inputs) + (epsilon / K)
