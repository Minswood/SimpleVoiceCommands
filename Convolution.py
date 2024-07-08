import numpy as np
from numpy import asarray
import os
import csv


def ReLU(x):
    x = np.maximum(0.0, x)
    return x

def convolution(image, kernel):
    
    #Flip kernel 180
    kernel = np.flipud(np.fliplr(kernel))

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

# This function applies all saved filters on a single image, and saves their output. It calls the applySingleFilter function.


def applyAllFilters():
    InputCounter = 1
    FilterCounter = 1
    Channel = 1
    #channelCounter = 1
    Output = 1
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
            filterOutput = np.array([])
            Input = f'./Conv1/Conv1_Output/filterOutput{InputCounter}.csv' #Check the Output files location
            
           # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
            with open(Input,newline='')as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    filterOutput = np.append(filterOutput, row)
                filterOutput = filterOutput.reshape(30,30)
            
            filter = np.array([[]])
            FilterFile = f'./filters_conv2/channel{InputCounter}_filter{FilterCounter}.csv'
           # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
            with open(FilterFile,newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    filter = np.append(filter,row)
                filter = filter.reshape(3,3)
                
            new_image = convolution(filterOutput, filter)
            
            #Copying image shape to Oimage to add all the other filters to it
            if InputCounter == 1:
                Outputimage = new_image.copy()
                Outputimage -= new_image
                
            summedImage = addNextImage(Outputimage, new_image)
             
            InputCounter += 1
                 
            if(InputCounter == 33):

                # Adding the corresponding bias to all values in the filtered image and then calling the ReLu function on it
                for row in range(summedImage.shape[0]):
                    for column in range(summedImage.shape[1]):
                        summedImage[row,column] += biases[FilterCounter-1]
                        
                
                #ReLU        
                summedImage = ReLU(summedImage)
                
                FilterCounter += 1
                
                Outputimage = np.array([])
                np.savetxt(f'{output_directory}/Conv2Output{Output}.csv',summedImage)
                Output +=1
                summedImage = np.array([])
        
                InputCounter = 1
                
                if(FilterCounter == 65):
                    channelCounter +=1
                    FilterCounter = 1
              
                
        except:
            print("Exit filter loop at channel: "+ str(InputCounter) + " & filter: " + str(FilterCounter))
            break

#applyAllFilters()
