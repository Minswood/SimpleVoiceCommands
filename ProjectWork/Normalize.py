import numpy as np

"""
A function that does the same thing as the normalization layer of keras.
Normalization means going through every single element of every batch, and using the formula
(input - mean)/sqrt(var)
Mean is the  mean of the elements in batch and var is their variance

"""


def NormalizeSingle(input):
    
    #Defining variables and arrays. 
    output = np.zeros(shape=(np.shape(input)))
    varTable = np.zeros(shape=(np.shape(input)))
    #print('NormalizeSingle Input Shape:\n',input.shape)
    shape = input.shape[0]*input.shape[1]
    mean = 0
    var=0
    
    #Counting the mean of the input array
    for x in input:
        for y in x:
            mean += y
    mean = mean/(input.shape[0]*input.shape[1])
    #print('NormalizeSingle Input Mean:\n',mean)
    
    #Counting the variance of the input array
    for row in range(len(input)):
        for column in range(len(input[row])):
            varTable[row, column] = (input[row,column]-mean)**2        
    for x in varTable:
        for y in x:
            var +=y
    var = var/shape
    SD = var**0.5
    #print('NormalizeSingle Input Variance:\n',var)
    #print('NormalizeSingle Input SD:\n',SD)
   
    #The normalization of each element in input array and putting them into output array
    for row in range(len(input)):
        for column in range(len(input[row])):
            output[row,column] = (input[row,column]-mean)/(var**0.5)
        
    return output



# Calling the NormalizeSingle function for each batch in a layer. Turns out this one is not required, but it is still here just in case
def NormalizeLayer(layer):
    output = np.zeros(shape=np.shape(layer))
    for batch in range(len(layer)):
        #print('Workign Batch:\n',batch)
        output[batch]= NormalizeSingle(layer[batch])
    

    print('Length:',len(layer))
    return output








