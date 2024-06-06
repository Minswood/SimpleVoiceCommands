import numpy as np

def maxPooling2D(convoluted_img, pooling_size, stride):
    row = int((convoluted_img.shape[0] - pooling_size) / stride + 1)
    col = int((convoluted_img.shape[1] - pooling_size) / stride + 1)
    output_pooled_img = np.zeros(shape=(row, col))

    for i in range(row):
        for j in range(col):
            current = convoluted_img[i*stride : i*stride+pooling_size, j*stride : j*stride+pooling_size]
            output_pooled_img[i, j] = np.max(current)
            
    return output_pooled_img

##Testing

img = np.array([[3, 1, 1, 3], [2,5,0,2], [1,4,2,1], [4, 7, 2, 4]])
output=pooling2d(img, 2,2)

print("Image", img)
print("Pooling",output)
print("===============")
print(img.shape)
print(output.shape)
print("===============")
print("Full image" + "\n" + str(img))
print("maxPooled array " + "\n" + str(output))

"""
Output
------
Image [[3 1 1 3]
 [2 5 0 2]
 [1 4 2 1]
 [4 7 2 4]]
Pooling [[5. 3.]
 [7. 4.]]
===============
(4, 4)
(2, 2)
===============
Full image
[[3 1 1 3]
 [2 5 0 2]
 [1 4 2 1]
 [4 7 2 4]]
maxPooled array 
[[5. 3.]
 [7. 4.]]
"""
