import numpy as np  
import random

# Setting a random seed to allow better tracking
np.random.seed(42)

def relu_activation(x):
    return max(0, x)

def generate_weights_and_bias(input_size, units):
    weights = 0.1 * np.random.rand(input_size, units)
    bias = np.zeros((1, units))
    return weights, bias

# units specifies the number of neurons in the layer
def dense_1(input, units):
    weights, bias = generate_weights_and_bias(len(input), units)
    output = []
    for i in range(weights.shape[1]):
        product = 0
        for j in range(weights.shape[0]):
            product += input[j] * weights[j][i] 

        product += bias[0][i]
        product = relu_activation(product)
        output.append(product)

    print("output ", output)
    return np.array(output)

def main():
    input = np.array([0., 1., 2., 3., 
                    3., 4., 5., 6.,
                    5., 8., 9., 7.])
    
    dense_1(input, 8)

if __name__ == '__main__':
    main()