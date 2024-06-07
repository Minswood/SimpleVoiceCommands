import numpy as np #Used for random and arange commands


"""
Petteri Karjalainen OAMK 2024

Calculating dense function from keras without library usage.
output is: output = activation(dot(input, kernel) + bias) 
activation is voluntary of usage if command is used

relu activation function is called rectified linear unit
Dot is dot product calculation
Input is input data
Kernel is weight matrix data
Bias is bias data
random value is between 0-1 generated for testing

Change random_values and arrange_data for real functions.
"""


class dense: #Dense function class

    def relu_activation(x): #Relu activation in python
        return max(0.0, x)
    
    def dot_function(a, b): #Calculating dot function
        return a@b 

    def random_value(): #Random float generation for testing purpose
        return np.random.uniform(low=0.0, high=1.0)
    
    def arrange_data(size_stop): #Arange initilaizing shape
        return np.arange(start=0, stop=size_stop, step=1, dtype=np.float16)
        
#First dense layer data

input_data_1 = dense.arrange_data(128) #Input data
kernel_data_1 = dense.arrange_data(128) #Kernel data so called weights matrix
bias_data_1 = dense.random_value() #Bias data tells how far neuron is from right


#Generate random values for data    
for x in range(0,128,1):
    kernel_data_1[x] = dense.random_value()
    input_data_1[x] = dense.random_value()


print("Kernel data: ", kernel_data_1)
print("Input data: ", input_data_1)

"""
Command is:
output = activation(dot(input, kernel) + bias)
"""
#First dense layer
print("dense first layer is",dense.relu_activation((dense.dot_function(input_data_1, kernel_data_1) + bias_data_1)))


#Second dense layer data

input_data_2 = dense.arrange_data(64) #Input data
kernel_data_2 = dense.arrange_data(64) #Kernel data so called weights matrix
bias_data_2 = dense.random_value() #Bias data tells how far neuron is from right


#Generate random values for data    
for x in range(0,64,1):
    kernel_data_2[x] = dense.random_value()
    input_data_2[x] = dense.random_value()


print("Kernel data2: ", kernel_data_2)
print("Input data2: ", input_data_2)

"""
second dense for size labels command is:
output = dot(input, kernel) + bias
"""

#Second dense layer
print("dense second layer is", (dense.dot_function(input_data_2, kernel_data_2) + bias_data_2))