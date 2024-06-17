import math
import numpy as np

def bilinear_interpolation(arr, x, y):
    height, width = arr.shape
    x1 = max(min(math.floor(x), width - 1), 0)
    y1 = max(min(math.floor(y), height - 1), 0)
    x2 = max(min(math.ceil(x), width - 1), 0)
    y2 = max(min(math.ceil(y), height - 1), 0)

    dx = x - x1
    dy = y - y1

    p11 = float(arr[y1, x1])
    p12 = float(arr[y2, x1])
    p21 = float(arr[y1, x2])
    p22 = float(arr[y2, x2])

    interpolated = (
        p11 * (1 - dx) * (1 - dy) +
        p21 * dx * (1 - dy) +
        p12 * (1 - dx) * dy +
        p22 * dx * dy
    )
    return interpolated

def resize(image_array, new_height, new_width):
    input_array = np.asarray(image_array)
    print(input_array.shape)
    input_array = input_array[:, :, 0]
    height = input_array.shape[0]
    width = input_array.shape[1]
    x_center = (width - 1) / 2
    y_center = (height - 1) / 2
    
    x_new_center = (new_width - 1) / 2
    y_new_center = (new_height - 1) / 2

    x_scale = width / new_width
    y_scale = height / new_height

    output_array = np.zeros((new_height, new_width), input_array.dtype)
    
    for y in range(new_height):
        for x in range(new_width):
            p_x = (x - x_new_center) * x_scale + x_center
            p_y = (y - y_new_center) * y_scale + y_center

            output_array[y, x] = bilinear_interpolation(input_array, p_x, p_y)
    
    output_array = output_array[..., np.newaxis]
    print(output_array.shape)
    return output_array