import numpy as np




def Normalize(input):
    output = np.zeros(shape=(np.shape(input)))
    shape = input.shape[0]*input.shape[1]
    print(shape)
    mean = 0
    for x in input:
        for y in x:
            mean += y
    mean = mean/(input.shape[0]*input.shape[1])
    print(mean)
    for row in range(len(input)):
        for column in range(len(input[row])):
            output[row,column] = input[row,column]/90
        
    return output

picture = np.array([[22,33,90],
                    [54,7,50],
                    [77,56,13]])
newPicture = Normalize(picture)
#print(newPicture)
