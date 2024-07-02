import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import cv2

#Testing
""""
img = cv2.imread('Doge.png', cv2.IMREAD_GRAYSCALE) / 255
plt.imshow(img, cmap = 'gray')
plt.show()
img.shape
""""




def ReLU(x):
    x = np.maximum(0.0, x)
    return x


def convolution(image, kernel, padding = 0, strides=1):
    
    #Flip kernel 180
    kernel = np.flipud(np.fliplr(kernel))

     #Get image & kernel shape                 
    X_Kernel = kernel.shape[0]
    Y_Kernel = kernel.shape[1]
    X_Img = image.shape[0]
    Y_Img = image.shape[1]

    X_Output = ((X_Img - X_Kernel + 2 * padding) // strides) + 1
    Y_Output = (( Y_Img - Y_Kernel + 2 * padding) // strides) + 1
   
    Output = np.zeros(shape=(X_Output, Y_Output))
    
    for i in range(X_Output):
        for j in range(Y_Output):
            current = img[i : i + X_Kernel, j : j + Y_Kernel] 
            multiplication = (sum(sum(current * kernel)))
            Output[i, j] = multiplication
    return Output

#Using modified Kasperi's applyALLFilters code to go through all channels
# This function applies all saved filters on a single image, and saves their output. It calls the applySingleFilter function.

def applyAllFilters(image):
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
            filter = np.array([[]])
            FilterFile = f'./filters_conv2/channel{channelCounter}_filter{counter}.csv'
           # Appending the rows in one filter to the empty filter array and then applying it on the image using the applySingleFilter function
            with open(FilterFile,newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    filter = np.append(filter,row)
                filter = filter.reshape(3,3)
                #print('Filter',filter)
            new_image = convolution(image, filter)
        
            #Copying image shape to Oimage to add all the other filters to it
            if Channel == 1:
                Oimage = new_image.copy()
                print("==========",Oimage)
            Oimage += new_image
            print("==========2",Oimage)
            # Adding the corresponding bias to all values in the filtered image and then calling the ReLu function on it
            for row in range(Oimage.shape[0]):
                for column in range(Oimage.shape[1]):
                    Oimage[row,column] += biases[counter-1]
        
            #ReLU        
            OImage = ReLU(Oimage)
        
            counter += 1
            #print(counter)
            #print(channelCounter)
            #print(Channel)
            Channel = 0
            if(counter == 65):
                Channel = 1
                Oimage = np.array([])
                np.savetxt(f'{output_directory}/filter2Output{channelCounter}.csv',OImage)
                channelCounter +=1
                counter = 1
                
        except:
            print("Exit filter loop at channel: "+ str(channelCounter) + " & counter: " + str(counter))
            break


#applyAllFilters(img)
