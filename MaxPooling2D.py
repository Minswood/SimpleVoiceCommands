import numpy as np
import math

def maxPool2D(convoluted_img, *pool_size, strides):
    if len(pool_size) < 2: #Checks if pool_size has 2 values
        row = math.floor((convoluted_img.shape[0] - pool_size[0]) / strides) + 1
        col = math.floor((convoluted_img.shape[1] - pool_size[0]) / strides) + 1
        """ 
        Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window 
        (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension.
        from: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D 
        """

    if len(pool_size) >= 3: # If pool_size exceeds more than 3, it prints an error
        return print("Pool_size only accepts int or tuple of 2 integers. If only one integer is specified, the same window length will be used for all dimensions.")
    else:
        output_shape_row = math.floor((convoluted_img.shape[0] - pool_size[0]) / strides) + 1
        output_shape_col = math.floor((convoluted_img.shape[1] - pool_size[1]) / strides) + 1
    
    output_pooled_img = np.zeros(shape=(output_shape_row, output_shape_col)) #Creates an array with zeros in the size of row & col
    
    if len(pool_size) < 2:
        for i in range(row):
            for j in range(col):
                current = convoluted_img[i*strides : i*strides+pool_size[0], j*strides : j*strides+pool_size[0]]
                output_pooled_img[i, j] = np.max(current)
    else:
        for i in range(output_shape_row):
            for j in range(output_shape_col):
                current = convoluted_img[i*strides : i*strides+pool_size[0], j*strides : j*strides+pool_size[1]]
                output_pooled_img[i, j] = np.max(current)
            
    return output_pooled_img


"""
Testing
img = np.array([[3, 1, 1, 3], 
                [2, 5, 0, 2], 
                [1, 4, 2, 1], 
                [4, 7, 2, 4]])

output=maxPool2D(img, 2,2, strides=2)
print(output)
[[5. 3.]
 [7. 4.]]
"""
