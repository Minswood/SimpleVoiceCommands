import numpy as np


"""
Petteri Karjalainen OAMK 2024

Calculating dense function from keras without keras library usage.
output is: output = activation(dot(input, kernel) + bias) 
activation is voluntary of usage if command is used

activation function is relu
Dot is dot product calculation
Input is input data
Kernel is weight matrix data
Bias is bias data

"""


class dense: #Dense function class

    def relu_function(x): #Relu function in python it is 0.0 or lower
        return max(0.0, x)
    
    def dot_function(a, b): #Calculating dot function
        return a@b 
        
#Test input data
input_data = np.arange(128, dtype=np.uint8) #Input data
kernel_data = np.arange(128, dtype=np.uint8) #Kernel data so called weights matrix
bias_data = 20.2 #Bias data tells how far neuron is from right

"""
Command is:
output = activation(dot(input, kernel) + bias)
"""

print(dense.relu_function((dense.dot_function(input_data, kernel_data) + bias_data)))
