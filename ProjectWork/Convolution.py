import numpy as np
import csv
import os

'''
def applySingleFilter(input, filter):
    output = np.zeros(shape=np.shape(input))
    for rows in range(len(input)):
        for columns in range(len(input)):
            output[rows,columns]= (input[rows-1,columns-1]*filter[rows-1,columns-1])+(input[rows-1,columns]*filter[rows-1,columns])+(input[rows-1,columns+1]*filter[rows-1,columns+1])+(input[rows,columns-1]*filter[rows,columns-1])+(input[rows,columns]*filter[rows,columns])+(input[rows,columns+1]*filter[rows,columns+1])+(input[rows+1,columns-1]*filter[rows+1,columns-1])+(input[rows+1,columns]*filter[rows+1,columns])+(input[rows+1,columns+1]*filter[rows+1,columns+1])
            
'''

# The Convolution layer uses ReLu activation, which means all values below 0 are changed to 0. 
def ReLu(filtered):
    output = filtered
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            if output[row,column] < 0:
                print('Column before',output[row,column])
                output[row,column]=0
                print('Column after', output[row,column])
                
    return output

# This function iterates a single filter over an input image. It will be used in another function that calls it on multiple filters
def applySingleFilter(input, filter):
    output = np.zeros(shape=np.shape(input))
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
    counter=1
    directory = f'Conv1_Output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    while 1:
        try:
            filter = np.array([[]])
            FilterFile = f'./filters_conv1/channel1_filter{counter}.csv'
            with open(FilterFile,newline='')as csvfile:
                csvReader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    filter = np.append(filter,row)
                filter = filter.reshape(3,3)
                print('Filter',filter)
            new_image = applySingleFilter(image, filter)
            np.savetxt(f'{directory}/filterOutput{counter}.csv',new_image)
            counter += 1
        except:
            print('Exit filter loop at counter:',counter)
            break
'''
def applyAllFilters(image):
        counter = 1
        filter = np.array([[]])
        FilterFile = f'./filters_conv1/channel1_filter{counter}.csv'
        with open(FilterFile,newline='')as csvfile:
            csvReader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csvReader:
                filter = np.append(filter,row)
            filter = filter.reshape(3,3)
            print('Filter',filter)
        applySingleFilter(image, filter)
        counter += 1
        

        return
'''



#output[rows,columns] = (input[rows,columns]*filter[0,0])+(input[rows,columns+1]*filter[0,1])+(input[rows,columns+2]*filter[0,2])+(input[rows+1,columns]*filter[1,0])+(input[rows+1,columns+1]*filter[1,1])+(input[rows+1,columns+2]*filter[1,2])+(input[rows+2,columns]*filter[2,0])+(input[rows+2,columns+1]*filter[2,1])+(input[rows+2,columns+2]*filter[2,2])




filter = np.array([[-1,-1,-1],
                   [0,0,0],
                   [1,0.001,1]])  
picture = np.random.rand(64,64)

applyAllFilters(picture)


'''
filter = np.array([[-1,-1,-1],
                   [0,0,0],
                   [1,1,1]])  
print(picture)
print(filter)
newPic = applySingleFilter(picture,filter)
newPic = ReLu(newPic)
print('Filtered Picture:\n',newPic)
np.savetxt('FilteredImage.csv',newPic, delimiter=',')
np.savetxt('OriginalImage.csv',picture, delimiter=',')
'''
