import numpy as np
import csv

'''
def applySingleFilter(input, filter):
    output = np.zeros(shape=np.shape(input))
    for rows in range(len(input)):
        for columns in range(len(input)):
            output[rows,columns]= (input[rows-1,columns-1]*filter[rows-1,columns-1])+(input[rows-1,columns]*filter[rows-1,columns])+(input[rows-1,columns+1]*filter[rows-1,columns+1])+(input[rows,columns-1]*filter[rows,columns-1])+(input[rows,columns]*filter[rows,columns])+(input[rows,columns+1]*filter[rows,columns+1])+(input[rows+1,columns-1]*filter[rows+1,columns-1])+(input[rows+1,columns]*filter[rows+1,columns])+(input[rows+1,columns+1]*filter[rows+1,columns+1])
            
'''

# The Convolution layer used ReLu activation, which means all values below 0 are changed to 0. 
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
            
            output[rows,columns] = (input[rows,columns]*filter[0,0])+(input[rows,columns+1]*filter[0,1])+(input[rows,columns+2]*filter[0,2])+(input[rows+1,columns]*filter[1,0])+(input[rows+1,columns+1]*filter[1,1])+(input[rows+1,columns+2]*filter[1,2])+(input[rows+2,columns]*filter[2,0])+(input[rows+2,columns+1]*filter[2,1])+(input[rows+2,columns+2]*filter[2,2])
    return(output)



# This function applies all saved filters on a single image, and saves their output. It calls the applySingleFilter function.
def applyAllFilters(image, pathToFilters):
    filter = []
    FilterFile = f'c:\CodeStuff\SummerProject2024\ProjectWork\SimpleVoiceCommands\\filters_conv1\channel1_filter1'
    with open(FilterFile,newline='')as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            filter.append(row)
            print('Filter row',row)



    return


#C:\CodeStuff\SummerProject2024\ProjectWork\SimpleVoiceCommands\filters_conv1








filter = np.array([[-1,-1,-1],
                   [0,0,0],
                   [1,1,1]])  
picture = np.random.rand(64,64)

applyAllFilters(picture,'ff')

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
