import numpy as np
import math
import os
import csv

def maxPool2D(*pool_size, strides):
   
    maxPoolingOutput = np.array([[[]]])
    
    for InputCounter in range(64):
        #Get conv1 outputs
        conv2Output = np.array([])
        Input = f'./Conv2_Output/Conv2Output{InputCounter+1}.csv' #Check the Output files location
                
        # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
        with open(Input,newline='')as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                conv2Output = np.append(conv2Output, row)
            conv2Output = conv2Output.reshape(28,28)
    
        
        if len(pool_size) < 2: #Checks if pool_size has 2 values
            row = math.floor((conv2Output.shape[0] - pool_size[0]) / strides) + 1
            col = math.floor((conv2Output.shape[1] - pool_size[0]) / strides) + 1
            
            """ 
            Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window 
            (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension.
            from: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D 
            """
    
        if len(pool_size) >= 3: # If pool_size exceeds more than 3, it prints an error
            return print("Pool_size only accepts int or tuple of 2 integers. If only one integer is specified, the same window length will be used for all dimensions.")
        else:
            output_shape_row = math.floor((conv2Output.shape[0] - pool_size[0]) / strides) + 1
            output_shape_col = math.floor((conv2Output.shape[1] - pool_size[1]) / strides) + 1
        
        output_pooled_img = np.zeros(shape=(output_shape_row, output_shape_col)) #Creates an array with zeros in the size of row & col
        
        if len(pool_size) < 2:
            for i in range(row):
                for j in range(col):
                    current = conv2Output[i*strides : i*strides+pool_size[0], j*strides : j*strides+pool_size[0]]
                    current = np.asarray(current, dtype='float64')
                    output_pooled_img[i, j] = np.max(current)
                    
        else:
            for i in range(output_shape_row):
                for j in range(output_shape_col):
                    current = conv2Output[i*strides : i*strides+pool_size[0], j*strides : j*strides+pool_size[1]]
                    current = np.asarray(current, dtype='float64')
                    output_pooled_img[i, j] = np.max(current)
        print(output_pooled_img.shape)
        maxPoolingOutput = np.append(maxPoolingOutput, output_pooled_img)
             
    return  maxPoolingOutput.reshape(-1,14,14)
    
#output = maxPool2D(2,2, strides = 2)
#print(output.shape)    
