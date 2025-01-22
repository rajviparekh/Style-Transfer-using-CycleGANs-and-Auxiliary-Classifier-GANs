import os
import math
import pickle
import numpy as np
import tensorflow_datasets as tfds
# import umap.umap_ as umap
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, initializers, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# import tensorflow_addons as tfa
from data_utils import *
from model_opt import *
from model_arch import *





def load_model(parent_directory, folder_name, generator_a2b, generator_b2a, discriminator_a, discriminator_b, num_styles, gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer, generator_loss, discriminator_loss, aux_class_loss, cycle_loss, identity_loss):
    
    
    
    # Loading the Epoch Losses disctionary
    with open(parent_directory+folder_name+'/epoch_losses.pickle', 'rb') as f:
            epoch_losses = pickle.load(f)
            
            
    # Creating an AC Cycle GAN model using the class created above
    model = CycleGANWithAuxClassifier(generator_a2b, generator_b2a, discriminator_a, discriminator_b, num_styles, lambda_aux=2)

    # Compiling the AC Cycle GAN
    model.compile(
    gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer, generator_loss, discriminator_loss, aux_class_loss, cycle_loss, identity_loss,)

    # Calling the model with some sample input to initialize the variables
    dummy_input = (tf.zeros([1, 128, 128, 3]), tf.zeros([1, ]))
    _ = model(dummy_input,training=False)

    # Loading model weights
    model.load_weights(parent_directory+folder_name+'/best_generator_a2b_loss.h5')

    return epoch_losses, model
         
            

def plot_losses(epoch_losses):
    # Creating the figure and axis objects for the plot
    fig,ax = plt.subplots(figsize=(12,7))

    # Setting the grid properties
    ax.grid(color='#adadad',axis='both',which='major',linestyle='--',alpha=0.5,zorder=-1)

    # Setting the properties of the plot border
    plt.setp(ax.spines.values(), linewidth=2, color='k')

    epochs = range(1, len(epoch_losses["gen_p2a_loss"]) + 1)

    # Plot the losses for each epoch
    plt.plot(epochs, epoch_losses["gen_a2b_loss"], lw=3.5, label="Photo to Art Generator Loss")
    plt.plot(epochs, epoch_losses["gen_b2a_loss"], lw=3.5, label="Art to Photo Generator Loss")
    plt.plot(epochs, epoch_losses["disc_a_loss"], lw=3.5, label="Photo Discriminator Loss")
    plt.plot(epochs, epoch_losses["disc_b_loss"], lw=3.5, label="Art Discriminator Loss")
    plt.plot(epochs, epoch_losses["aux_class_a_loss"], lw=3.5, label="Photo Auxiliary Classifier Loss")
    plt.plot(epochs, epoch_losses["aux_class_b_loss"], lw=3.5, label="Art Auxiliary Classifier Loss")
    plt.plot(epochs, epoch_losses["cycle_a_loss"], lw=3.5, label="Photo Cycle Loss")
    plt.plot(epochs, epoch_losses["cycle_b_loss"], lw=3.5, label="Art Cycle Loss")
    plt.plot(epochs, epoch_losses["identity_a_loss"], lw=3.5, label="Photo Identity Loss")
    plt.plot(epochs, epoch_losses["identity_b_loss"], lw=3.5, label="Art Identity Loss")

    # Setting the properties of the ticks on both the axes
    ax.tick_params(axis='both', which='major',labelcolor='k',labelsize=14, length=0)

    # Setting the labels for the x and y axes
    ax.set_xlabel('Epochs',fontsize=18)
    ax.set_ylabel('Losses',fontsize=18)

    # Plotting the legend
    legend = ax.legend(handleheight=1,handlelength=2)
    # Setting up the legend properties
    legend.set_title('Type of Loss')
    legend.get_title().set_fontsize(14)
    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_facecolor('None')
    [text.set_fontsize(12) for text in legend.get_texts()]

    # Setting the title
    ax.set_title('Loss Plot',fontsize=22)

    plt.show()



# Function to plot a multiple photos alongside translated versions of both art-styles using the AC Cycle GAN model
def plot_generated_images(model, dataset, num_images=6, num_columns=6, separation=True):

    # Calculate the number of rows needed based on the number of images and columns
    num_rows = (num_images * 3) // num_columns

    if (num_images * 3) % num_columns != 0:
        num_rows += 1

    # Get a batch of images from the dataset
    data = next(iter(dataset))

    # Randomly select a subset of images from the batch
    idx = tf.random.shuffle(tf.range(data.shape[0]))[:num_images]
    image_batch = tf.gather(data, idx)

    # Generate Ukiyo-e and Monet-style images for the batch
    ukiyo_e_images = model.predict((image_batch, tf.ones((image_batch.shape[0]))))
    monet_images = model.predict((image_batch, tf.zeros((image_batch.shape[0]))))

    # Create a plot with the specified number of rows and columns
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 10))

    # Loop over each row and column in the plot and add the corresponding image
    for row in range(num_rows):
        for col in range(num_columns):

            index = (row * num_columns + col) // 3
            style = (row * num_columns + col) % 3

            if col % 3 == 0:
                # Original image
                axs[row, col].imshow((image_batch[index] + 1) / 2)
                if row == 0:
                    axs[row, col].set_title('Ordinary Pictures',fontsize=18)
            elif col % 3 == 1:
                # Ukiyo-e style generated image
                axs[row, col].imshow((monet_images[index] + 1) / 2)
                if row == 0:
                    axs[row, col].set_title('Monet versions',fontsize=18)
            elif col % 3 == 2:
                # Monet style generated image
                axs[row, col].imshow((ukiyo_e_images[index] + 1) / 2)
                if row == 0:
                    axs[row, col].set_title('Ukiyo-e versions',fontsize=18)

            axs[row, col].axis('off')

    # Redraw the canvas to update the plot
    fig.canvas.draw()

    # Add a vertical line between the two subplot groups
    if separation:
        line = fig.add_artist(plt.Line2D([0.5, 0.5], [0, 0.95], color='k', lw=3, ls='--'))

    # Add a title to the plot and adjust the layout
    fig.suptitle("Original Photographs and Generated Paintings", y=1, fontsize=28, fontweight='bold')
    fig.tight_layout()

    # Show the plot
    plt.show()
