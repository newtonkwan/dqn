# Predict the sine function using pure JAX. 
# Date created: April 13, 2022
# Last updated: April 14, 2022 

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import hydra
import optax
import haiku as hk
import os
import pickle
from pathlib import Path
from typing import Union


def forward(batch: jnp.ndarray) -> jnp.ndarray:
    '''Standard MLP network'''
    mlp = hk.nets.MLP(output_sizes=[128, 128, 1], w_init = hk.initializers.RandomNormal(stddev=1e-2), b_init = hk.initializers.RandomNormal(stddev=1e-2), name="standard_mlp")
    return mlp(batch)

def generate_sin(num_examples, batch_size, key):
    '''Generates samples from sin function'''
    key, init_key = jax.random.split(key)
    x_train = jax.random.uniform(init_key, shape=(num_examples, batch_size, 1), minval=0, maxval=2*np.pi)
    y_train = jnp.sin(x_train)
    x_test = jax.random.uniform(key, shape=(num_examples//10, batch_size, 1), minval=0, maxval=2*np.pi)
    y_test = jnp.sin(x_test)
    return x_train, y_train, x_test, y_test

# Save and load functions from https://github.com/google/jax/issues/2116
def save(data, path: Union[str, Path], overwrite: bool = False):
    '''Save function'''
    suffix = '.pickle'
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load(path: Union[str, Path]):
    '''Load function'''
    suffix = '.pickle'
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

@hydra.main(config_path="conf", config_name="config")
def main(args): 
    # Make the network and optimizer 
    init_net, apply_net = hk.without_apply_rng(hk.transform(forward))
    optimizer = optax.adam(learning_rate = args.learning_rate)

    def loss(params, batch: jnp.ndarray, labels: jnp.ndarray):
        '''Loss function'''
        y_hat = apply_net(params, batch)
        loss_value = optax.l2_loss(y_hat, labels).mean() # MSE
        return loss_value

    @jax.jit
    def step(params, opt_state, batch, labels):
        '''Calculate gradients and update parameters'''
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    def training_loop(params, optimizer, x_train, y_train, x_test, y_test, num_epochs):
        '''Training loop'''
        opt_state = optimizer.init(params)
        train_losses, test_losses = [], []

        num_batches = len(x_train) 
        for epoch in range(num_epochs):
            epoch_sum_of_train_loss = 0 
            start_time = time.time()
            for batch_i in range(0, num_batches):    
                params, opt_state, step_train_loss = step(
                    params, 
                    opt_state,
                    x_train[batch_i], 
                    y_train[batch_i])
                epoch_sum_of_train_loss += step_train_loss
            epoch_time = time.time() - start_time

            # calculate train loss 
            epoch_train_loss = epoch_sum_of_train_loss/len(x_train) # avg train loss per batch per epoch
            train_losses.append(epoch_train_loss)

            # calculate test loss 
            flattened_x_test = x_test.reshape(-1, 1)
            flattened_y_test = y_test.reshape(-1, 1)
            epoch_test_loss = loss(params, flattened_x_test, flattened_y_test)
            test_losses.append(epoch_test_loss)

            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training loss {}".format(epoch_train_loss))
            print("Test loss {}".format(epoch_test_loss))
        return params, train_losses, test_losses

    def plot(network_params, x_train, y_train, x_test, train_losses, test_losses, epochs):
        '''Show plots'''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[1.75*6.4, 1*4.8])
        ax1.scatter(x_train, y_train, s=3, label="Training data")
        ax1.scatter(x_test, apply_net(network_params, x_test), s=3, label = "Model Prediction")
        ax1.set_title("Sin function")
        ax1.legend()
        ax2.plot(range(epochs), train_losses, label="Training loss")
        ax2.plot(range(epochs), test_losses, label = "Test loss")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.set_title("Training and Test Losses")
        path = os.path.dirname(os.path.abspath(__file__))
        print("Saving plot ... ")
        plt.savefig(path+'/plots/haiku_plots.png', bbox_inches='tight', dpi=400)
        print(f"Done. Path to plot: {path+'/plots/haiku_plots.png'}")
        plt.show()

    # Initailize rng keys 
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    # Generate training data
    x_train, y_train, x_test, y_test = generate_sin(args.num_batches, args.batch_size, init_key)

    # Initialize the network; note we draw an input to get shapes
    sample_input = jax.random.uniform(key, shape=(1, 1), minval=0, maxval=2*np.pi)
    # Can't include layer sizes unless i want to pass in layer sizes every call of apply_net()
    # layer_sizes = [args.layers.layer1_size, 
    #             args.layers.layer2_size, 
    #             args.layers.output_size]
    network_params = init_net(key, sample_input)

    # Train model 
    network_params, train_loss, test_loss = training_loop(
        network_params, 
        optimizer,
        x_train, 
        y_train, 
        x_test, 
        y_test,
        args.epochs)

    # Save model 
    path = os.path.dirname(os.path.abspath(__file__))
    model_name = "sine_haiku_model"
    print("Saving model ...")
    print(f"Done. Path to model: {path}/models/{model_name}.pickle")
    save(network_params, path=path+"/models/"+model_name, overwrite=True)

    # Show plots 
    plot(network_params, x_train, y_train, x_test, train_loss, test_loss, args.epochs) 

    # Load model 
    '''
    filename = model_name+".pickle"
    loaded_params = load(path+"/models/"+filename)
    '''

if __name__ == "__main__":
    main() 