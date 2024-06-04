import numpy as np
import math



def Normalize(input):
    output = np.zeros(shape=(np.shape(input)))
    varTable = np.zeros(shape=(np.shape(input)))
    shape = input.shape[0]*input.shape[1]
    print('input:', input)
    mean = 0
    biggest = 0
    smallest = 100
    var=0
    for x in input:
        for y in x:
            mean += y
    mean = mean/(input.shape[0]*input.shape[1])
    print('mean:',mean)
    for row in range(len(input)):
        for column in range(len(input[row])):
            varTable[row, column] = (input[row,column]-mean)**2
            print(('Mistake:',input[row,column]-mean))
    print('varTable:',varTable)
    for x in varTable:
        for y in x:
            var +=y
    var = var/shape
    print('var:',var)
    print(biggest)
    print(smallest)
    print(var)
    for row in range(len(input)):
        for column in range(len(input[row])):
            output[row,column] = (input[row,column]-mean)/(var**0.5)
        
    return output

picture = np.array([[22,33,90],
                    [54,7,50],
                    [77,56,13]])
newPicture = Normalize(picture)


picture2 = np.array([[47,33,66],
                    [54,20,50],
                    [61,56,13]])
newPicture2 = Normalize(picture2)
print('Result:\n',newPicture)
print('Result:\n',newPicture2)
