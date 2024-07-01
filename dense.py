import numpy as np  
import random
import csv

import scipy

def relu_activation(x):
    return max(0, x)
    
def get_weights(path_weights):
    weights = []
    try:
        with open(path_weights, newline='') as csvfile:
            result = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for weight in result:
                weights.append(weight)
    except FileNotFoundError as e:
        print("get_weights:", str(e))
    else:
        weights = np.asarray(weights)
        return weights
    
def get_biases(path_biases):
    biases = []
    try:
        with open(path_biases, newline='') as csvfile:
            result = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for bias in result:
                biases.append(bias[0])
    except FileNotFoundError as e:
        print("get_biases:", str(e))
    else:
        biases = np.asarray(biases)
        return biases

def get_weights_and_biases(path_weights, path_biases):
    weights = get_weights(path_weights)
    biases = get_biases(path_biases)
    return weights, biases

# units specifies the number of neurons (outputs) in the layer
def dense_1(input, units):
    weights, bias = get_weights_and_biases('dense1Weights.csv', 'dense1Biases.csv')
    output = []
    try:
        for i in range(units):
            product = 0
            for j in range(input.size):
                product += input[j] * weights[j][i] 
            product += bias[i]
            product = relu_activation(product)
            output.append(product)
    except TypeError as e:
        print("Type error: " + str(e))
    except IndexError as e:
        print("Index error: " + str(e))
    except Exception as e:
        print(str(e))
    else:
        output = np.array(output)
        print("dense1 output shape", output.shape)
        return output

def dense_2(input, units):
    weights, bias = get_weights_and_biases('dense2Weights.csv', 'dense2Biases.csv')
    output = []
    try:
        for i in range(units):
            product = 0
            for j in range(input.size):
                product += input[j] * weights[j][i] 
            product += bias[i]
            output.append(product)
    except TypeError as e:
        print("Type error: " + str(e))
    except IndexError as e:
        print("Index error: " + str(e))
    except Exception as e:
        print(str(e))
    else:
        output = np.array(output)
        # Convert output to propabilities that amount to one.
        output = np.exp(output) / sum(np.exp(output)) 
        print("dense2 output shape", output.shape)
        return output

def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    predictions = []

    # Random input data for testing
    input = np.random.rand(12544,)
    
    dense1 = dense_1(input, 128)
   
    if(dense1 is not None):
        predictions = dense_2(dense1, len(labels))
        print("Predictions ", predictions)


if __name__ == '__main__':
    main()