import numpy as np  
import random
import csv
import scipy
import tensorflow as tf

def relu_activation(x):
    return max(0.0, x)
    
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
        weights = np.asarray(weights, dtype=np.float32)
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
        biases = np.asarray(biases, dtype=np.float32)
        return biases

def get_weights_and_biases(path_weights, path_biases):
    weights = get_weights(path_weights)
    biases = get_biases(path_biases)
    return weights, biases

def dense_1(input):
    # units specifies the number of neurons (outputs) in the layer
    units = 128
    weights, bias = get_weights_and_biases('dense1Weights.csv', 'dense1Biases.csv')
    output = []
    try:
        # product = np.dot(input, weights) + bias
        # for i in range (units):
        #     activated = relu_activation(product[i])
        #     output.append(activated)

        # product = np.dot(input, weights) + bias
        for i in range(units):
            product = 0
            counter = 0
            for j in range(input.size):
                product += input[j] * weights[j][i] 
                # if(i == 4 and product <1 ):
                #     print("product from loop", product)
                #     counter +=1
                # if(i == 4):
                #     print("counter", counter)
           
            product += bias[i]
            product = relu_activation(product)
        # print("dense 1, product shape", output.shape)
            output.append(product)
        
    except TypeError as e:
        print("Type error: " + str(e))
    except IndexError as e:
        print("Index error: " + str(e))
    except Exception as e:
        print(str(e))
    else:
        output = np.array(output)
        # print("dense 1 output", output)
        # print("dense1 output shape", output.shape)
        return output

def dense_2(input):
    units = 8
    weights, bias = get_weights_and_biases('dense2Weights.csv', 'dense2Biases.csv')
    output = []
  
    try:
        # output = np.dot(input, weights) + bias
        # for i in range (units):
        #     activated = relu_activation(product[i])
        #     output.append(activated)

        # product = np.dot(input, weights) + bias
        for i in range(units):
            product = 0
            dense2counter = 0
            for j in range(input.size):
                product += input[j] * weights[j][i] 
                dense2counter += 1
            # print("Product before bias", product)
            product += bias[i]
            # print("Product after bias", product)

            # product = relu_activation(product)
            
        # print("dense 2, product shape", output.shape)
            output.append(product)
            # print('Dense counter', dense2counter)
    except TypeError as e:
        print("Type error: " + str(e))
    except IndexError as e:
        print("Index error: " + str(e))
    except Exception as e:
        print(str(e))
    else:
        # print("tf softmax", tf.nn.softmax(output))
        # Convert output to propabilities that amount to one.
        # output = np.exp(output) / sum(np.exp(output)) 
        # output = getPercent(output)
        return output
    
def getPercent(input):
    sum = 0
    output = []
    for x in input:
        sum += x
    for x in input:
        output.append((x/sum) * 100)
    return output

def main():
    labels = ["Down", "Go", "Left", "No", "Right", "Stop", "Up", "Yes"]
    predictions = []

    # Random input data for testing
    input = np.random.uniform(size=12544)
    
    dense1 = dense_1(input)
   
    if(dense1 is not None):
        predictions = dense_2(dense1)
        print("Predictions ", predictions)


if __name__ == '__main__':
    main()