import numpy as np

def flatten(input_list: np.array) -> np.array:
    flattened = np.array([[[]]])

    for sublist in input_list: 
        for sublist2 in sublist:
            for element in sublist2:
                flattened = np.append(flattened,element)

    return flattened 

#print(flatten(output).shape)
