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
 
    #Flip kernel 180
    # kernel = np.flipud(np.fliplr(kernel))

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

def convReady(image,kernel):
    kernel = kernel[::-1,::-1]
    convolved = cv2.filter2D(image, -1, kernel) # Convolve
 
    H = np.floor(np.array(kernel.shape)/2).astype(np.int32) # Find half dims of kernel
    convolved = convolved[H[0]:-H[0],H[1]:-H[1]] # Cut away unwanted information    
 
 
    return  convolved

# This function applies all saved filters on a single image, and saves their output. It calls the applySingleFilter function.
def Conv1(image):
    counter = 1
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
    
    # The combination of while and try-except is used, so that the function goes through all filter files one at a time, and stops when done
    while 1:
        try:
            filter = np.array([[]])
            FilterFile = f'./filters_conv1/channel1_filter{counter}.csv'
 
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
                    new_image[row,column] += biases[counter-1]
            
            new_image = ReLu(new_image)
            # Saving the output to the created directory as csv files.
            np.savetxt(f'{output_directory}/filterOutput{counter}.csv',new_image,delimiter=',')
            counter += 1
        except:
            print('Exit filter loop at counter:',counter)
            break
 
    return

