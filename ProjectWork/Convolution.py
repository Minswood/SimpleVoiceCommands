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
def applySingleFilter(input, filter):
    output = np.zeros(shape=(np.shape(input)[0]-2,np.shape(input)[0]-2))
    for rows in (range(len(input)-2)):
        for columns in (range(len(input)-2)):
            element = 0
            for f_row in range(len(filter)):
                for f_column in range(len(filter)):
                    element += filter[f_row,f_column] * input[(rows+f_row),(columns+f_column)]
            output[rows,columns] = element
                              
    return(output)



# This function applies all saved filters on a single image, and saves their output. It calls the applySingleFilter function.
def applyAllFilters(image):
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
                print('Filter',filter)
            new_image = applySingleFilter(image, filter)
  
            
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



#This is for testing purposes


filter = np.array([[-1,-1,-1],
                   [0,0,0],
                   [1,0.001,1]])  
picture = np.random.rand(32,32)

applyAllFilters(picture)
