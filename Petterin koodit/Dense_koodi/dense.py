import numpy as np


"""
Petteri Karjalainen OAMK 2024

Calculating dense function from keras without keras library usage.
output is: output = activation(dot(input, kernel) + bias) 
activation function is relu
"""

Batch_size = 2 #Input batch size
Input_dim = 20 #Input dim

class dense:

    def relu_function(x): #Relu function in python it is 0.0 or higher
        return max(0.0, x)
    
    def dot_function(a, b): #Calculating dot function 
        return np.dot(a,b)
    
    outputarray = np.empty(128, dtype=np.double)
    
inputdata= np.empty(128, dtype=np.double) #Empty numpy array for test

test_data = 200
kernel_data = 200
bias_data = 20
#inputneurons[0:2] = 20 #How to edit neurons data
#print(inputneurons) #Relu function test print

print(dense.relu_function((dense.dot_function(test_data, kernel_data) + bias_data)))
