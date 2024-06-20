import numpy as np  
import random
import csv
import matplotlib.pyplot as plt
import tensorflow as tf

def relu_activation(x):
    return max(0, x)

def get_weights(path_weights):
    weights = []
    with open(path_weights, newline='') as csvfile:
        result = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for weight in result:
            weights.append(weight)
        weights = np.asarray(weights)
    return weights

def get_biases(path_biases):
    biases = []
    with open(path_biases, newline='') as csvfile:
        result = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for bias in result:
            biases.append(bias[0])
        biases = np.array(biases)
    return biases

def get_weights_and_biases(path_weights, path_biases):
    weights = get_weights(path_weights)
    biases = get_biases(path_biases)

    return weights, biases

# units specifies the number of neurons (outputs) in the layer
def dense_1(input, units):
    weights, bias = get_weights_and_biases('dense1Weights.csv', 'dense1Biases.csv')
    
    output = []
    for i in range(units):
        product = 0
        for j in range(input.size):
            product += input[j] * weights[j][i] 

        product += bias[i]
        product = relu_activation(product)
        output.append(product)

    output = np.array(output)
    print("dense1 output shape", output.shape)
    return output

def dense_2(input, units):
    weights, bias = get_weights_and_biases('dense2Weights.csv', 'dense2Biases.csv')
    
    output = []
    for i in range(units):
        product = 0
        for j in range(input.size):
            product += input[j] * weights[j][i] 

        product += bias[i]
        output.append(product)

    output = np.array(output)
    print("dense2 output shape", output.shape)
    return output

def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]

    input = np.random.rand(12544,)    
    
    dense1 = dense_1(input, 128)
    predictions = dense_2(dense1, len(labels))
    print("Predictions ", predictions)


if __name__ == '__main__':
    main()