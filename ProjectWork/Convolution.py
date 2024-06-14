import numpy as np

'''
def applySingleFilter(input, filter):
    output = np.zeros(shape=np.shape(input))
    for rows in range(len(input)):
        for columns in range(len(input)):
            output[rows,columns]= (input[rows-1,columns-1]*filter[rows-1,columns-1])+(input[rows-1,columns]*filter[rows-1,columns])+(input[rows-1,columns+1]*filter[rows-1,columns+1])+(input[rows,columns-1]*filter[rows,columns-1])+(input[rows,columns]*filter[rows,columns])+(input[rows,columns+1]*filter[rows,columns+1])+(input[rows+1,columns-1]*filter[rows+1,columns-1])+(input[rows+1,columns]*filter[rows+1,columns])+(input[rows+1,columns+1]*filter[rows+1,columns+1])
            
'''




def applySingleFilter(input, filter):
    output = np.zeros(shape=np.shape(input))
    for rows in (range(len(input)-2)):
        for columns in (range(len(input)-2)):
            
            output[rows,columns] = (input[rows,columns]*filter[0,0])+(input[rows,columns+1]*filter[0,1])+(input[rows,columns+2]*filter[0,2])+(input[rows+1,columns]*filter[1,0])+(input[rows+1,columns+1]*filter[1,1])+(input[rows+1,columns+2]*filter[1,2])+(input[rows+2,columns]*filter[2,0])+(input[rows+2,columns+1]*filter[2,1])+(input[rows+2,columns+2]*filter[2,2])
            print('HAHAHAHA',rows,columns)
    return(output)










filter = np.array([[-1,-1,-1],
                   [0,0,0],
                   [1,1,1]])  
picture = np.arange(4096).reshape((64,64))

print(picture)
print(filter)
newPic = applySingleFilter(picture,filter)
print('Filtered Picture:\n',newPic)
