import numpy as np
import pickle # Python module to serialize and deserialize Python objects, which is used to read the CIFAR-10 data files.
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import keras_tuner as kt


def load_cifar_batch(filename):
  # Opening the file in binary read mode.
  with open(filename, 'rb') as f: 
    data_dict = pickle.load(f, encoding='bytes') # loading the data, which is stored as bytes, from the file

    # retrieving image data from the loaded dictionary. data is a flat array of pixel values
    # prefixing a string with b creates a byte literal. b'data' is a byte string representing the ASCII characters "data."
    data = data_dict[b'data']  

    labels = data_dict[b'labels'] # retrieving the labels corresponding to the images.

    # Reshaping the flat data array into a 4D array where each image has dimensions 3x32x32 (3 color channels, 32x32 pixels)
    # and transposing the array dimensions to be (num_samples, height, width, channels), the format expected by TensorFlow.
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')

    labels = np.array(labels) # Converts the labels to a NumPy array for easy manipulation

  return data, labels

# loading training batches
x_train, y_train = [], []
for i in range(1, 6):
  data, labels = load_cifar_batch(f'/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment8/cifar-10-batches-py/data_batch_{i}')
  x_train.append(data)
  y_train.append(labels)

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

# loading test batch
x_test, y_test = load_cifar_batch('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment8/cifar-10-batches-py/test_batch')

# Normalize data, scales the pixel values to [0, 1], which helps in speeding up convergence during training
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Building the CNN Model Using the Functional API
# input_shape : shape of the input data (32, 32, 3) for CIFAR-10 (image dimensions (32x32 pixels) and no. of color channels (3 for RGB))
# num_classes : number of output classes
# num_filters : Specifies the number of filters (also known as kernels) in each convolutional layer. Filters are responsible for 
#               extracting features from the input data. More filters can capture more complex features but also increase computational cost.
# kernel_size : Dimensions of the filter used in the convolutional layers. Integer or Tuple of two integers — 3, (3, 3), 5. 
#             Common sizes are 3x3 or 5x5. Smaller kernels capture fine details, while larger ones capture more global patterns.
# num_conv_layers : number of convolutional layers in the network. More layers -> network learns more abstract features at each 
#                 level of the hierarchy, but increasing layers -> increased computational cost / risk of overfitting.
# pool_size : Dimensions of the pooling window used in the max-pooling layers. Integer or Tuple of two integers — 2, (2,2), 3. 
#           Pooling reduces the spatial dimensions (width and height) of the feature maps, reducing the number of parameters and computational cost.
# dropout_rate : The dropout rate specifies the fraction (0.2, 0.5, etc.) of input units to drop during training to prevent overfitting. Dropout randomly 
#              sets a fraction of the input units to zero at each update during training time, which helps in regularizing the model.
# regularization : Eg. 0.001, 0.0001. The L2 regularization factor applied to the weights of the convolutional layers. Regularization helps 
#                prevent overfitting by adding a penalty term to the loss function that discourages complex models with large weights.
# optimizer : The optimizer determines how the model weights are updated based on the loss function. Popular optimizers include 
#           Adam and RMSprop, which are adaptive learning rate methods that adjust the learning rate during training.
def create_cnn_model(input_shape, num_classes, num_filters, kernel_size, num_conv_layers,
                     pool_size, dropout_rate, regularization, optimizer):
  # Defines the input layer with the specified shape for CIFAR-10 images (32x32 pixels, 3 color channels)
  inputs = Input(shape=input_shape)
  # Initializes x with the input layer, starting the layer chaining process.
  x = inputs 

  # Layer Chaining: Each layer transformation updates x, and this updated x is used as input to the next layer. 
  # This pattern allows for flexible and dynamic model architectures.
  for _ in range(num_conv_layers):
    x = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu',
               padding='same', kernel_regularizer=l2(regularization)) (x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    x = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu',
               padding='same', kernel_regularizer=l2(regularization)) (x)
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)
    x = Dropout(dropout_rate)(x)

  # Flatten Layer: Converts the 2D matrix of feature maps into a 1D vector, preparing it for the fully connected layer.
  x = Flatten()(x)
  # Dense Layer (Fully Connected)
  x = Dense(512, activation='relu')(x)
  # Another dropout is applied to further reduce overfitting
  x = Dropout(dropout_rate)(x)
  # output layer
  outputs = Dense(num_classes, activation='softmax')(x)

  # model compilation
  # Instantiates the model with the defined input and output tensors, constructing the neural network.
  model = Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
  return model

# function to create the CNN model with tunable hyperparameters, using KerasTuner to search for the optimal hyperparameter configuration.
# takes a HyperParameters object (hp) and returns a compiled Keras model. 
# To be used by the tuner to build and test models with different hyperparameter configurations.
def model_builder(hp):
  model = create_cnn_model(input_shape=(32, 32, 3), num_classes=10,
                            num_filters=hp.Int('num_filters', min_value=32, max_value=128, step=32),
                            kernel_size=hp.Choice('kernel_size', values=[3, 5]),
                            num_conv_layers=hp.Int('num_conv_layers', min_value=2, max_value=4, step=1),
                            pool_size=hp.Choice('pool_size', values=[2, 3]),
                            dropout_rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1),
                            regularization=hp.Float('regularization', min_value=1e-5, max_value=1e-3, sampling='LOG'),
                            optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']))
  return model

# RandomSearch tuner randomly samples the hyperparameter space to find the best model configuration.
# max_trials=20: 20 different hyperparameter configurations will be tested.
# executions_per_trial=2: Each trial is executed twice and the results averaged (accounts for variability in model training)
# directory='path', project_name='cnn_tuning': Specify the directory and project name for storing tuning results and logs.
tuner = kt.RandomSearch(model_builder, objective='val_accuracy',
                        max_trials=10, executions_per_trial=1,
                        directory='/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment8/keras_tuner', project_name='cnn_tuning')

# Training the Model and Tuning Hyperparameters
tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Obtaining optimal hyperparameters
# tuner explores different combinations of hyperparameters specified in the model_builder function, using the RandomSearch strategy.
# For each trial, the tuner trains a model with a specific configuration, evaluates it on the validation set, and records the validation accuracy.
# 
# num_trials=1: Specifies that we want to retrieve the top trial (the single best-performing configuration).
# get_best_hyperparameters() returns a list of HyperParameters objects. Each object represents a set of hyperparameters for a 
# trial that achieved a high score based on the specified objective metric (val_accuracy in this case).
# Even with num_trials=1, the method returns a list containing the top result(s), so [0] is used to access the first element 
# of this list, which corresponds to the single best-performing set of hyperparameters.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# best_hps.get('param_name'): Retrieves the value of the specified hyperparameter from the best_hps object.
print(f"Best hyperparameters:\n"
      f"Number of filters: {best_hps.get('num_filters')}\n"
      f"Kernel size: {best_hps.get('kernel_size')}\n"
      f"Number of Conv layers: {best_hps.get('num_conv_layers')}\n"
      f"Pool size: {best_hps.get('pool_size')}\n"
      f"Dropout rate: {best_hps.get('dropout_rate')}\n"
      f"Regularization: {best_hps.get('regularization')}\n"
      f"Optimizer: {best_hps.get('optimizer')}")






    







