
import tensorflow as tf
import keras
import numpy as np

the_model = tf.keras.models.load_model('C:/CodeStuff/SummerProject2024/model_export.keras')#Use the directory where you saved the model.

the_model.summary()#Model information

layer1_weights = the_model.layers[1].get_weights()[0]
print(layer1_weights)
print('shape:',np.shape(layer1_weights))
#Trying to get the weights of specific layers.


np.savetxt('weights1.csv',layer1_weights, delimiter=',')#Writing the weights to a csv file to be used in our own functions