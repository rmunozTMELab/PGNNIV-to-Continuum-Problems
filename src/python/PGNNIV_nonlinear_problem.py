import os
import time
import random
import argparse
import pickle
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import keras

import gc
import psutil
from memory_profiler import profile

from sklearn.model_selection import train_test_split

custom_seed = 42
np.random.seed(custom_seed)
tf.random.set_seed(custom_seed)

# Load data from pickle file
def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Check directory and if it doesn't exist, create it
def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"The directory '{directory_path}' has been created.")
    else:
        print(f"The directory '{directory_path}' already exists.")


MODEL_NAME = "nonlinear_problem_P3"
data_filename = "../../data/" + MODEL_NAME + "_data.pkl"
data_abs_path = os.path.abspath(data_filename)
data = load_data_from_pickle(file_path=data_abs_path)
x_step_size = data['x_step_size']
y_step_size = data['y_step_size']
# ---------------------------------------------------------

# Finite difference convolutional operator to derivate in x axis
def Dx(f, x_step_size=x_step_size):
    Dx = tf.constant([[-1, +1], 
                      [-1, +1]], 
                     dtype=tf.float32)/(2*x_step_size)

    f_reshaped = tf.expand_dims(f[:, :, :], axis=-1)    
    Dx = tf.expand_dims(tf.expand_dims(Dx, axis=-1), axis=-1)
    dfdx = tf.nn.conv2d(f_reshaped, Dx, strides=[1, 1, 1, 1], padding='VALID', name='dfdx')
    return tf.constant(tf.squeeze(dfdx, axis=-1), tf.float32)

# Finite difference convolutional operator to derivate in y axis
def Dy(f, y_step_size=y_step_size):
    Dy = tf.constant([[+1, +1], 
                      [-1, -1]],  
                     dtype=tf.float32)/(-2*y_step_size)

    f_reshaped = tf.expand_dims(f[:, :, :], axis=-1)    
    Dy = tf.expand_dims(tf.expand_dims(Dy, axis=-1), axis=-1)
    dfdy = tf.nn.conv2d(f_reshaped, Dy, strides=[1, 1, 1, 1], padding='VALID', name='dfdy')
    return tf.constant(tf.squeeze(dfdy, axis=-1), tf.float32)

# Convolutional operator to do the mean between two elements of a mesh in x axis
def Mx(f):
    Mx = tf.constant([[+1, +1]], 
                     dtype=tf.float32)/(2)

    f_reshaped = tf.expand_dims(f[:, :, :], axis=-1)    
    Mx = tf.expand_dims(tf.expand_dims(Mx, axis=-1), axis=-1)
    x_avg = tf.nn.conv2d(f_reshaped, Mx, strides=[1, 1, 1, 1], padding='VALID', name='Mx')
    return tf.constant(tf.squeeze(x_avg, axis=-1), tf.float32)

# Convolutional operator to do the mean between two elements of a mesh in y axis
def My(f):
    My = tf.constant([[+1], 
                      [+1]], 
                     dtype=tf.float32)/(2)

    f_reshaped = tf.expand_dims(f[:, :, :], axis=-1)    
    My = tf.expand_dims(tf.expand_dims(My, axis=-1), axis=-1)
    y_avg = tf.nn.conv2d(f_reshaped, My, strides=[1, 1, 1, 1], padding='VALID', name='My')
    return tf.constant(tf.squeeze(y_avg, axis=-1), tf.float32)

# MSE as defined in the paper: sum of all the squarers of all the components of a tensor
def MSE(diff_tensor):
    return tf.reduce_sum(tf.square(diff_tensor))

# Constraint e
def e_constraint(y_true, y_pred):
    return y_true - y_pred

# Constraint pi1
def pi1_constraint(X_true, f_true, y_pred, K):
    K = K
    dydx_pred = Dx(y_pred)
    dydy_pred = Dy(y_pred)
    qx_pred = tf.math.multiply(K, (dydx_pred))
    qy_pred = tf.math.multiply(K, (dydy_pred))
    return  (-(Dx(qx_pred) + Dy(qy_pred)) - Mx(Mx(My(My(f_true)))))

# Constraint pi2
def pi2_constraint(X_true, y_pred):
    return tf.concat([tf.expand_dims(y_pred[:, :, 0], axis=1) - tf.expand_dims(X_true[:, :, 0], axis=1),
                      tf.expand_dims(y_pred[:, :, -1], axis=1) - tf.expand_dims(X_true[:, :, 1], axis=1), 
                      tf.expand_dims(y_pred[:, 0], axis=1) - tf.expand_dims(X_true[:, :, 2], axis=1),
                      tf.expand_dims(y_pred[:, -1], axis=1) - tf.expand_dims(X_true[:, :, 3], axis=1),
                      ], axis=1)

# Constraint pi3
def pi3_constraint(X_true, y_pred, K):
    dydx_pred = Dx(y_pred)
    dydy_pred = Dy(y_pred)

    qx_pred = tf.math.multiply(K, (dydx_pred))
    qy_pred = tf.math.multiply(K, (dydy_pred))

    X_true_red = My(X_true)
    return tf.concat([tf.expand_dims(qx_pred[:, :, 0], axis=1) - tf.expand_dims(X_true_red[:, :, 4], axis=1),
                      tf.expand_dims(qx_pred[:, :, -1], axis=1) - tf.expand_dims(X_true_red[:, :, 5], axis=1), 
                      tf.expand_dims(qy_pred[:, 0], axis=1) - tf.expand_dims(X_true_red[:, :, 6], axis=1),
                      tf.expand_dims(qy_pred[:, -1], axis=1) - tf.expand_dims(X_true_red[:, :, 7], axis=1),
                      ], axis=1)


@tf.keras.utils.register_keras_serializable()
class NeuralNetwork(tf.keras.Model):

    def __init__(self, input_size, hidden1_dim, hidden2_dim, output_size, n_filters=15, **kwargs):
        super(NeuralNetwork, self).__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_size = output_size
        self.n_filters = n_filters

        # Predictive network
        self.flatten_layer_pred = tf.keras.layers.Flatten(name="flatten_layer_pred")
        self.hidden1_layer_pred = tf.keras.layers.Dense(hidden1_dim, activation='sigmoid', name="hidden1_layer_pred")
        self.hidden2_layer_pred = tf.keras.layers.Dense(hidden2_dim, activation='sigmoid', name="hidden2_layer_pred")
        self.output_layer_pred = tf.keras.layers.Dense(output_size[0] * output_size[1], activation=None, name="output_layer_pred")

        # Explanatory network
        self.conv1_exp = tf.keras.layers.Conv2D(n_filters, (1, 1), activation='sigmoid', name="conv1_exp")
        self.flatten_layer_exp = tf.keras.layers.Flatten(name="flatten_layer_exp")
        self.hidden1_layer_exp = tf.keras.layers.Dense(hidden1_dim, activation='sigmoid', name="hidden1_layer_exp")
        self.hidden2_layer_exp = tf.keras.layers.Dense(hidden2_dim, activation='sigmoid', name="hidden2_layer_exp")
        self.output_layer_exp = tf.keras.layers.Dense(n_filters * (output_size[0] - 1) * (output_size[1] - 1), name="output_layer_exp")
        self.conv2_exp = tf.keras.layers.Conv2D(1, (1, 1), activation=None, name="conv2_exp")

    def call(self, X):
        
        # Predictive network
        X_pred_flat = self.flatten_layer_pred(X)
        X_pred_hidden1 = self.hidden1_layer_pred(X_pred_flat)
        X_pred_hidden2 = self.hidden2_layer_pred(X_pred_hidden1)
        output_dense_pred = self.output_layer_pred(X_pred_hidden2)

        # Average operator along x and y directions
        u_pred = tf.reshape(output_dense_pred, [tf.shape(output_dense_pred)[0], self.output_size[0], self.output_size[1]])
        um_pred = Mx(My(u_pred))

        # Explanatory network
        conv_output_exp = self.conv1_exp(tf.expand_dims(um_pred, axis=-1))
        conv_output_flat_exp = self.flatten_layer_exp(conv_output_exp)
        X_exp_hidden1 = self.hidden1_layer_exp(conv_output_flat_exp)
        X_exp_hidden2 = self.hidden2_layer_exp(X_exp_hidden1)
        output_exp = self.output_layer_exp(X_exp_hidden2)
        output_exp_reshaped = tf.reshape(output_exp, (um_pred.shape[0], um_pred.shape[1], um_pred.shape[2], self.n_filters))
        K_pred = tf.squeeze(self.conv2_exp(output_exp_reshaped), axis=-1)

        return u_pred, K_pred
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "input_size": self.input_size,
            "hidden1_dim": self.hidden1_dim,
            "hidden2_dim": self.hidden2_dim,
            "output_size": self.output_size,
            "n_filters": self.n_filters,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def train_neural_network(model, optimizer, X_train, y_train, f_train, num_epochs=100, batch_size=64):

    # Check if GPUs are available
    if tf.config.list_physical_devices('GPU'):
        print("\nTraining on GPU\n")
        device = '/gpu:0'
    else:
        print("\nTraining on CPU\n")
        device = '/cpu:0'

    train_total_loss_list = []
    train_total_MSE_list = []
    train_e_loss_list = []
    train_pi1_loss_list = []
    train_pi2_loss_list = []
    train_pi3_loss_list = []

    test_total_loss_list = []
    test_total_MSE_list = []
    test_e_loss_list = []
    test_pi1_loss_list = []
    test_pi2_loss_list = []
    test_pi3_loss_list = []

    # Train/test split
    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X_train, y_train, f_train, test_size=0.3, random_state=42)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    f_train = tf.convert_to_tensor(f_train, dtype=tf.float32)
    f_test = tf.convert_to_tensor(f_test, dtype=tf.float32)

    # Neural network training loop
    with tf.device(device):
        epoch = 0
        for epoch in range(num_epochs):
            for batch_start in range(0, len(X_train), batch_size):
                X_batch = X_train[batch_start:batch_start+batch_size]
                y_batch = y_train[batch_start:batch_start+batch_size]
                f_batch = f_train[batch_start:batch_start+batch_size]

                with tf.GradientTape(persistent=False) as tape:

                    # Forward pass
                    predictive_output, explanatory_output = model.call(X_batch)
                    predictions = predictive_output
                    K = explanatory_output

                    # Loss term computing
                    e = e_constraint(y_batch, predictions)
                    pi1 = pi1_constraint(X_batch, f_batch, predictions, K)
                    pi2 = pi2_constraint(X_batch, predictions)
                    pi3 = pi3_constraint(X_batch, predictions, K)

                    loss = 1e7 * MSE(e) + 1e4 * MSE(pi1) + 1e3 * MSE(pi2) + 1e5 * MSE(pi3)
                    sum_MSE_train = MSE(e) + MSE(pi1) + MSE(pi2) + MSE(pi3)

                # Gradient computing
                gradients = tape.gradient(loss, list(model.trainable_variables))
                optimizer.apply_gradients(zip(gradients, list(model.trainable_variables)))

            # Save losses in lists
            train_total_loss_list.append(loss.numpy())
            train_total_MSE_list.append(sum_MSE_train.numpy())
            train_e_loss_list.append(MSE(e).numpy())
            train_pi1_loss_list.append(MSE(pi1).numpy())
            train_pi2_loss_list.append(MSE(pi2).numpy())
            train_pi3_loss_list.append(MSE(pi3).numpy())

            # Test values
            test_predictive, test_explanatory = model.call(X_test)
            test_predictions = test_predictive
            test_K = test_explanatory

            test_e = e_constraint(y_test, test_predictions)
            test_pi1 = pi1_constraint(X_test, f_test, test_predictions, test_K)
            test_pi2 = pi2_constraint(X_test, test_predictions)
            test_pi3 = pi3_constraint(X_test, test_predictions, test_K)

            test_loss = 1e7 * MSE(test_e) + 1e4 * MSE(test_pi1) + 1e3 * MSE(test_pi2) + 1e5 * MSE(test_pi3)
            sum_MSE_test = MSE(test_e) + MSE(test_pi1) + MSE(test_pi2) + MSE(test_pi3)

            test_total_loss_list.append(test_loss.numpy())
            test_total_MSE_list.append(sum_MSE_test.numpy())
            test_e_loss_list.append(MSE(test_e).numpy())
            test_pi1_loss_list.append(MSE(test_pi1).numpy())
            test_pi2_loss_list.append(MSE(test_pi2).numpy())
            test_pi3_loss_list.append(MSE(test_pi3).numpy())

            if epoch % (1 if num_epochs < 100 else (10 if num_epochs <= 1000 else 100)) == 0:
                memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"Epoch {epoch}, Memory Usage: {memory_usage_mb:.2f} MB")
                print(f'Epoch {epoch}, Train loss: {loss.numpy()}, Test loss: {test_loss.numpy()}, sum(MSE) train: {sum_MSE_train.numpy()}, sum(MSE) test: {sum_MSE_test.numpy()}, MSE(e): {MSE(e)}, MSE(pi1): {MSE(pi1)}, MSE(pi2): {MSE(pi2)}, MSE(pi3): {MSE(pi3)}')

        print("\nTraining finished after", num_epochs, "epochs\n")

        results_dict = {
            'train_total_loss_list': train_total_loss_list,
            'train_total_MSE_list': train_total_MSE_list,
            'train_e_loss_list': train_e_loss_list,
            'train_pi1_loss_list': train_pi1_loss_list,
            'train_pi2_loss_list': train_pi2_loss_list,
            'train_pi3_loss_list': train_pi3_loss_list,
            'test_total_loss_list': test_total_loss_list,
            'test_total_MSE_list': test_total_MSE_list,
            'test_e_loss_list': test_e_loss_list,
            'test_pi1_loss_list': test_pi1_loss_list,
            'test_pi2_loss_list': test_pi2_loss_list,
            'test_pi3_loss_list': test_pi3_loss_list,
        }

    return results_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a machine learning model')
    parser.add_argument('epochs', type=int, help='Number of training epochs')
    parser.add_argument('resume_training', type=int, choices=[0, 1], help='Boolean to resume training (0 or 1)')

    args = parser.parse_args()
    args.model_name = "nonlinear_problem_P3" #ESTO ESTA PUESTO PROVISIONALMENTE PARA QUE NO DE PROBLEMAS. LA IDEA DE ESTO ES PASARLO COMO ARGUMENTO DEL PROGRAMA

    print("\n")
    print("Python script configuration:")
    print(" Epochs of traning:", args.epochs)
    print(" Resume training (0 == No, 1 == Yes):", args.resume_training)
    print(" Model name:", args.model_name)
    print("\n")
    
    # Define data path and results folder path
    data_filename = "../../data/" + args.model_name + "_data.pkl"
    data_abs_path = os.path.abspath(data_filename)
    data_dir_path = os.path.dirname(data_abs_path)
    
    results_folder = "../../results/" + args.model_name + "_results/"
    results_folder_path = os.path.abspath(results_folder)

    # Load data and create results folder
    data = load_data_from_pickle(file_path=data_abs_path)
    check_and_create_directory(directory_path=results_folder_path)
    
    # We load the variables from data we are going to use
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    f_train = data['f_train']
    f_val = data['f_val']

    n_data = data['n_data']
    n_discretization = data['n_discretization']
    x_step_size = data['x_step_size']
    y_step_size = data['y_step_size']

    print("\n")
    print("Forma de X_train:", X_train.shape)
    print("Forma de X_val:", X_val.shape)
    print("Forma de y_train:", y_train.shape)
    print("Forma de y_val:", y_val.shape)
    print("\n")

    # Load model
    input_shape = X_train[0].shape
    hidden1_dim = 150
    hidden2_dim = 150
    output_shape = y_train[0].shape

    print("\n")
    print("Neural network shape")
    print("----------------------")
    print("Neurons in the input layer:", input_shape)
    print("Neurons in the 1st hidden layer:", hidden1_dim)
    print("Neurons in the 2nd hidden layer:", hidden2_dim)
    print("Neurons in the output layer:", output_shape)
    print("\n")

    if args.resume_training and args.model_name:

        print("\n")
        print('Loading model ' + '"' + str(args.model_name) + '"')

        model_pretrained_weights_path = os.path.abspath(results_folder_path + "/" + args.model_name + "_first_train.weights.h5")
        model_pretrained_results_path = os.path.abspath(results_folder_path + "/" + args.model_name + "_first_train.pkl")

        model_new_weights_path = os.path.abspath(results_folder_path + "/" + args.model_name + "_new_train.weights.h5")
        model_new_results_path = os.path.abspath(results_folder_path + "/" + args.model_name + "_new_train.pkl")

        # Create model, call it to initialize and load weights
        model_loaded = NeuralNetwork(input_shape, hidden1_dim, hidden2_dim, output_shape)
        _ = model_loaded.call(X_train)
        model_loaded.load_weights(model_pretrained_weights_path)   

        # Load results from the pretrained model to get de K for the last training loop
        with open(model_pretrained_results_path, "rb") as f:
            data = pickle.load(f)   

        start_time = time.time()
        training = train_neural_network(model=model_loaded,
                             optimizer = tf.optimizers.Adam(learning_rate=3e-5),
                             X_train=X_train,
                             y_train=y_train,
                             f_train=f_train,
                             num_epochs=args.epochs,
                            #  batch_size=64,
                             )
        execution_time = time.time() - start_time
        print(f"Execution time of train_neural_network: {execution_time} seconds.")
        
        predictions_pred, predictions_exp = model_loaded.call(X_val)

        data = {
            'training': training,
            'predictions_pred': predictions_pred.numpy(), 
            'predictions_exp': predictions_exp.numpy(),
        }

        with open(model_new_results_path, "wb") as f:
            pickle.dump(data, f)

        model_loaded.save_weights(model_new_weights_path)

    else:
        print("\n")
        print("Training of the model from scratch.")
        print("\n")

        model_weights_path = os.path.abspath(results_folder_path + "/" + args.model_name + "_first_train.weights.h5")
        model_results_path = os.path.abspath(results_folder_path + "/" + args.model_name + "_first_train.pkl")

        model = NeuralNetwork(input_shape, hidden1_dim, hidden2_dim, output_shape)
        optimizer = tf.optimizers.Adam(learning_rate=3e-4)

        start_time = time.time()
        training = train_neural_network(model=model,
                             optimizer = optimizer,
                             X_train=X_train,
                             y_train=y_train,
                             f_train=f_train,
                             num_epochs=args.epochs,
                            #  batch_size=64,
                             )
        execution_time = time.time() - start_time
        print(f"Execution time of train_neural_network: {execution_time} seconds.")
        
        predictions_pred, predictions_exp = model.call(X_val)

        data = {
            'training': training,
            'predictions_pred': predictions_pred.numpy(), 
            'predictions_exp': predictions_exp.numpy(),
        }

        with open(model_results_path, "wb") as f:
            pickle.dump(data, f)

        model.save_weights(model_weights_path)

        
        









