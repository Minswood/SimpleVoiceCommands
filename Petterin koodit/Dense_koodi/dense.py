import numpy as np


"""
Petteri Karjalainen OAMK 2024

Calculating dense command from keras without keras library.

output is: output = activation(dot(input, kernel) + bias) 

activation is relu

"""

Batch_size = 2 
Input_dim = 20


class dense:

    def relu_function(x): #Relu function in python it is 0.0 or higher
        return max(0.0, x)
    
    outputarray = np.empty(128, dtype=np.double)
    



inputneurons = np.empty(128, dtype=np.double) #Empty numpy array for test

inputneurons[0:2] = 20 #How to edit neurons data
print(inputneurons) #Relu function test print
