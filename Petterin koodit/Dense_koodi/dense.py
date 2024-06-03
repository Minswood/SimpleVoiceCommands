import numpy as np #Used for numpy
import random #Used for random generation


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

    def relu_activation(x): #Relu function in python it is 0.0 or lower
        return max(0.0, x)
    
    def dot_function(a, b): #Calculating dot function
        return a@b 

    def random_value(): #Random float generation
        return np.random.uniform(low=0.0, high=1.0)
        
#Test input data
input_data = np.arange(start=0, stop=128, step=1, dtype=np.float16) #Input data
kernel_data = np.arange(start=0, stop=128, step=1, dtype=np.float16) #Kernel data so called weights matrix
bias_data = 20.2 #Bias data tells how far neuron is from right


#Generate random values for begining test
for x in range(0,128,1):
    input_data[x] = dense.random_value()
    
for x in range(0,128,1):
    kernel_data[x] = dense.random_value()


print("Kernel data: ", kernel_data)
print("Input data: ", input_data)

"""
Command is:
output = activation(dot(input, kernel) + bias)
"""

print(dense.relu_activation((dense.dot_function(input_data, kernel_data) + bias_data)))
