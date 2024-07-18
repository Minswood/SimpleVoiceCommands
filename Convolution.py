import cv2
import numpy as np
import csv
import os


# The Convolution layer uses ReLu activation, which means all values below 0 are changed to 0. 
def ReLu(filtered):
    output = filtered
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            if output[row,column] < 0:
                output[row,column]=0
                
    return output




# This function iterates a single filter over an input image. It will be used in another function that calls it on multiple filters
def applySingleFilter(image, kernel):
    kernel_size = len(kernel)
    row = image.shape[0] - len(kernel) + 1
    col = image.shape[1] - len(kernel[0]) + 1
    Output = np.zeros(shape=(row, col))

    for i in range(row):
        for j in range(col):
            current = image[i : i + kernel_size, j : j + kernel_size]
            current = np.asarray(current, dtype='float64')
            multiplication = (sum(sum(current * kernel)))
            Output[i, j] = multiplication
    return Output


# This function calls the apllySingleFilter to apply all filters in filters_conv1 folder to input image 
def Conv1(image):
    output_directory = f'Conv1_Output'
    #Saving the biases in a list for ease of use
    biases = []
    with open('./conv1Biases.csv',newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    biases.append(row)
 
    # Creating a directory for filter output if one does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    
    # A for loop that goes through all filters.
    for counter in range(32):
        try:
            filter = np.array([[]])
            FilterFile = f'./filters_conv1/channel1_filter{counter+1}.csv'
 
           # A single filter is opened and applied to the image:
            with open(FilterFile,newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    filter = np.append(filter,row)
                filter = filter.reshape(3,3)
                # print('Filter',filter)
                
            new_image = applySingleFilter(image, filter)
            # new_image = convReady(image,filter)
            
            # Adding the corresponding bias to all values in the filtered image and then calling the ReLu function on it
            for row in range(new_image.shape[0]):
                for column in range(new_image.shape[1]):
                    new_image[row,column] += biases[counter]
            new_image = ReLu(new_image)
            
            # Saving the output to the created directory as a csv file.
            np.savetxt(f'{output_directory}/filterOutput{counter+1}.csv',new_image,delimiter=',')
        except:
            print('Early exit in filter loop at iteration number:',counter+1)
            break
 
    return

