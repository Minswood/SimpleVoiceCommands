import cv2
import numpy as np
from numpy import asarray
import os
import csv


def ReLU(x):
    x = np.maximum(0.0, x)
    return x


def convolution(image, kernel):
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

def addNextImage(base, addition):
    try:
        if np.shape(base) == np.shape(addition):
            output = np.zeros(shape=(np.shape(base)))
            for row in range(len(base)):
                for column in range(len(base[row])):
                    output[row,column] = (base[row,column] + addition[row,column])
        else:
            print('Images not same shape')
    except:
        print('Error in addNextImage')
    return output


def Conv2():
    output_directory = f'Conv2_Output'
    #Saving the biases in an array for ease of use
    biases = []
    with open('./conv2Biases.csv',newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    biases.append(row)
 
    #Creating a directory for filter output if one does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # First for loop tells which filter is being used and second one tells which input image is being used.
    for FilterCounter in range(64):
        for InputCounter in range(32):
            #Get conv1 outputs
                inputImage = np.array([])
                Input = f'./Conv1_Output/filterOutput{InputCounter+1}.csv' #Check the Output files location
                
            # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
                with open(Input,newline='')as csvfile:
                    reader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                    for row in reader:
                        inputImage = np.append(inputImage, row)
                    inputImage = inputImage.reshape(30,30)
                
                filter = np.array([[]])
                FilterFile = f'./filters_conv2/channel{InputCounter+1}_filter{FilterCounter+1}.csv'
            # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
                with open(FilterFile,newline='')as csvfile:
                    csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                    for row in csvReader:
                        filter = np.append(filter,row)
                    filter = filter.reshape(3,3)
                    
                new_image = convolution(inputImage, filter)
                # new_image = convReady(filterOutput, filter)
                #Copying image shape to Oimage to add all the other filters to it
                if InputCounter == 0:
                    Outputimage = new_image.copy()
                    Outputimage -= new_image
                summedImage = addNextImage(Outputimage, new_image) 
                Outputimage = summedImage
        # InputCounter loop ends here
        
        # Adding biases to the summed image
        for row in range(summedImage.shape[0]):
            for column in range(summedImage.shape[1]):
                summedImage[row,column] += biases[FilterCounter]     
        #ReLU        
        summedImage = ReLU(summedImage)
        # Saving the summedImage to a csv file and reseting the array.
        np.savetxt(f'{output_directory}/Conv2Output{FilterCounter+1}.csv',summedImage, delimiter=',')
        summedImage = np.array([])

        
