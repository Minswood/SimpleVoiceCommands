import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('Doge.png', cv2.IMREAD_GRAYSCALE) / 255
plt.imshow(img, cmap = 'gray')
plt.show()
img.shape

kernel =  np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])


def relu(img: np.array) -> np.array:
    img = img.copy()
    img[img < 0] = 0
    return img
    
def convolution(image, kernel, padding = 0, strides=1):
    #kernel_size = len(kernel)
    #Flip kernel 180 to cross-correlate
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
    return relu(Output)

##### Testing and printing #####
output = convolution(img, kernel)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(output, cmap="gray")
plt.axis("off")

img

output
