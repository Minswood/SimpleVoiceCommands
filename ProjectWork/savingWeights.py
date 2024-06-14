
import tensorflow as tf
import csv
import numpy as np
import os

the_model = tf.keras.models.load_model('C:/CodeStuff/SummerProject2024/model_export.keras')#Use the directory where you saved the model.

the_model.summary()#Model information

#Saving all the weights and biases of required layers
conv1_filters = the_model.layers[2].get_weights()[0]
conv2_filters = the_model.layers[3].get_weights()[0]
conv1_biases = the_model.layers[3].get_weights()[1]
conv2_biases = the_model.layers[3].get_weights()[1]

dense1_weights = the_model.layers[7].get_weights()[0]
dense2_weights = the_model.layers[9].get_weights()[0]
dense1_biases = the_model.layers[7].get_weights()[1]
dense2_biases = the_model.layers[9].get_weights()[1]

#Shapes of the two filter layers
print('Shape1:\n',np.shape(conv1_filters),'\nShape2:\n',np.shape(conv2_filters))

# A Function to write filters to CSV in a filter directory. Each filter is written as their own file.
def filter_to_csv(layer_name, layer_filters):
    directory = f'filters_{layer_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filter_set in range(layer_filters.shape[-1]):
        
        for filter_channel in range(layer_filters.shape[-2]):
            filter = layer_filters[:,:, filter_channel, filter_set]
            #print('Filter values:\n', filter)
            filename=f'{directory}/channel{filter_channel+1}_filter{filter_set+1}.csv'
            with open(filename, mode='w',newline='') as file:
                writer = csv.writer(file)
                writer.writerows(filter)
                print('Wrote:', filename)
            



print(np.shape(conv1_biases),'AAND', np.shape(conv2_biases))
print(np.shape(dense1_weights),'AND',np.shape(dense2_weights))

np.savetxt('dense1Weights.csv',dense1_weights, delimiter=',')
np.savetxt('dense1Biases.csv',dense1_biases, delimiter=',')


np.savetxt('dense2Weights.csv',dense2_weights, delimiter=',')
np.savetxt('dense2Biases.csv',dense2_biases, delimiter=',')

filter_to_csv('conv1', conv1_filters)
filter_to_csv('conv2', conv2_filters)

np.savetxt('conv1Biases.csv',conv1_biases, delimiter=',')
np.savetxt('conv2Biases.csv',conv2_biases, delimiter=',')