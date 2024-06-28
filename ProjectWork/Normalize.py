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
    print('NormalizeSingle Input Shape:\n',input.shape)
    shape = input.shape[0]*input.shape[1]
    mean = 0
    var=0
    
    #Counting the mean of the input array
    for x in input:
        for y in x:
            mean += y
    mean = mean/(input.shape[0]*input.shape[1])
    print('NormalizeSingle Input Mean:\n',mean)
    
    #Counting the variance of the input array
    for row in range(len(input)):
        for column in range(len(input[row])):
            varTable[row, column] = (input[row,column]-mean)**2        
    for x in varTable:
        for y in x:
            var +=y
    var = var/shape
    print('NormalizeSingle Input Variance:\n',var)
   
    #The normalization for each element of input array and putting them into output array
    for row in range(len(input)):
        for column in range(len(input[row])):
            output[row,column] = (input[row,column]-mean)/(var**0.5)
        
    return output



# Calling the NormalizeSingle function for each batch in a layer.
def NormalizeLayer(layer):
    output = np.zeros(shape=np.shape(layer))
    for batch in range(len(layer)):
        #print('Workign Batch:\n',batch)
        output[batch]= NormalizeSingle(layer[batch])
    

    print('Length:',len(layer))
    return output



#Everyting beyond this points is for testing purposes and will be removed at a later date.
'''
test_layer = np.random.rand(64,32,32)
new_layer = NormalizeLayer(test_layer)
for i in new_layer:
    print(i)
'''


#print ('The random layer:\n',test_layer)
#print('The layer shape:\n',np.shape(test_layer))
#output = np.zeros(shape=np.shape(test_layer))
#print('TEST:\n',test_layer[1],'\nAND THE SHAPE:\n',np.shape(test_layer[1]))



'''
Counter = 0
for batch in test_layer:
    print('BATCH:\n',batch,'\nBATCH SHAPE:\n',np.shape(batch))
    Counter+=1
print('Batch Count:',Counter)
'''










'''
picture = np.array([[22,33,90],
                    [54,7,50],
                    [77,56,13]])
newPicture = NormalizeSingle(picture)


picture2 = np.array([[47,33,66],
                    [54,20,50],
                    [61,56,13]])
newPicture2 = NormalizeSingle(picture2)

print('Result:\n',newPicture)
print('Result:\n',newPicture2)

means = 0
for x in newPicture:
    for y in x:
        means += y
means = means/(newPicture.shape[0]*newPicture.shape[1])
print('mean:',means)
'''