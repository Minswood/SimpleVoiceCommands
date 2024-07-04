import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
import os
import csv
import cv2



def ReLU(x):
    x = np.maximum(0.0, x)
    return x

def convolution(image, kernel):
    
    #Flip kernel 180
    kernel = np.flipud(np.fliplr(kernel))

    kernel_size = len(kernel)
    row = img.shape[0] - len(kernel) + 1
    col = img.shape[1] - len(kernel[0]) + 1
    Output = np.zeros(shape=(row, col))

    for i in range(row):
        for j in range(col):
            current = img[i : i + kernel_size, j : j + kernel_size]
            multiplication = (sum(sum(current * kernel)))
            Output[i, j] = multiplication
    return Output


# This function applies all saved filters on a single image, and saves their output. It calls the applySingleFilter function.


def applyAllFilters():
    OutputCounter = 1
    counter = 1
    Channel = 1
    channelCounter = 1
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
    

    
    
    # The combination of while and try-except is used, so that the function goes through all filter files one at a time, and stops when done
    while 1:
        try:
            #Get conv1 outputs
            filterOutput = np.array([[]])
            OutputFiles = f'./Conv1/Conv1_Output/filterOutput{OutputCounter}.csv' #Check the Output files location
            #OutputFiles = f'./Conv1/filters_conv1/channel1_filter{OutputCounter}.csv' for testing
           # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
            with open(OutputFiles,newline='')as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    filterOutput = np.append(filterOutput, row)

                print(OutputCounter)
                print('FilterOutput',filterOutput)
         
            filter = np.array([[]])
            FilterFile = f'./filters_conv2/channel{channelCounter}_filter{counter}.csv'
           # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
            with open(FilterFile,newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    filter = np.append(filter,row)
                filter = filter.reshape(3,3)
                print('Filter',filter)
            new_image = convolution(filterOutput, filter)
            print(new_image)
            
            #Copying image shape to Oimage to add all the other filters to it
            if OutputCounter == 1:
                Oimage = new_image.copy()
                Oimage -= new_image
              
            Oimage += new_image

            OutputCounter += 1
                 
            print("Counter = ", counter) #Which filter is going through all the 32 outputs
            print("Current channel = ", channelCounter)
            if(OutputCounter == 33):

                 # Adding the corresponding bias to all values in the filtered image and then calling the ReLu function on it
                for row in range(Oimage.shape[0]):
                    for column in range(Oimage.shape[1]):
                        Oimage[row,column] += biases[counter-1]
                        
                print("Biases = ",biases[counter-1])
                
                #ReLU        
                OImage = ReLU(Oimage)
                
                counter += 1
                
                Oimage = np.array([])
                
                np.savetxt(f'{output_directory}/filter2Output{counter}Channel{channelCounter}.csv',OImage)
                
                OutputCounter = 1
                
                if(counter == 65):
                    channelCounter +=1
                    counter = 1
              
                
        except:
            print("Exit filter loop at channel: "+ str(channelCounter) + " & counter: " + str(counter))
            break


#applyAllFilters()
