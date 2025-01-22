import os
import io
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
from model_eval import *
from PIL import Image

import wandb



# Binary cross-entropy loss function for the generator
def generator_loss(generated_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_output, labels=tf.ones_like(generated_output)))

# Binary cross-entropy loss function for the discriminator
def discriminator_loss(real_output, generated_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output)))
    generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_output, labels=tf.zeros_like(generated_output)))
    return real_loss + generated_loss

# Sparse categorical cross-entropy loss function for the auxiliary classifier
def aux_class_loss(labels, logits):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

# L1 loss function for cycle consistency loss
def cycle_loss(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image - cycled_image))

# L1 loss function for identity loss
def identity_loss(real_image, same_image):
    return tf.reduce_mean(tf.abs(real_image - same_image))

# Create a function to make a learning rate schedule that decays exponentially starting from the specified epoch
def create_lr_schedule(initial_lr, decay_start_epoch, dataset, decay_rate=0.9, staircase=False):
    steps_per_epoch = len(dataset)
    decay_steps = decay_start_epoch * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, decay_steps, decay_rate, staircase=staircase,)
    return lr_schedule


def plot_images(data, num_images=25):
    # Determine the grid size
    num_rows = int(math.sqrt(num_images))
    num_cols = int(math.ceil(num_images / num_rows))


    # Check if the batch includes labels
    if isinstance(data, tuple):
        image_batch, label_batch = data
    else:
        image_batch = data
        label_batch = None

    # Randomly select a subset of images (and corresponding labels, if available)
    idx = tf.random.shuffle(tf.range(tf.shape(image_batch)[0]))[:num_images]
    image_batch = tf.gather(image_batch, idx)
    label_batch = tf.gather(label_batch, idx) if label_batch is not None else None

    # Normalize image colors to [0, 1]
    image_batch = (image_batch + 1) / 2

    # Plotting
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            ax.axis("off")
            continue
        
        img = image_batch[i].numpy()  # Convert to numpy array

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# Function to plot a single photo alongside its art-style translated versions using the AC Cycle GAN model
def plot_single_translation(data,model, epoch):
    
    # Define the folder path to save the weights and losses
    folder_path = os.path.join('Images', 'Training')

    # Create a new directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Getting a batch of images from the test dataset
    photo_batch = next(iter(data))

    # Shuffle the images in the batch randomly
    shuffled_batch = tf.random.shuffle(photo_batch)

    # Select a random image from the shuffled batch
    image = shuffled_batch[tf.random.uniform([], minval=0, maxval=shuffled_batch.shape[0], dtype=tf.int32)]

    # The input image and label are expanded to be compatible with the model input shape.
    image = tf.expand_dims(image, axis=0)
    monet_label = tf.expand_dims(0, axis=0)
    ukiyoe_label = tf.expand_dims(1, axis=0)


    # The model is used to predict the output image based on the input image and label.
    predicted_image_monet = model.predict((image, monet_label))
    predicted_image_ukiyoe = model.predict((image, ukiyoe_label))
    # The predicted image is converted from a numpy array to a format suitable for display.
    predicted_image_monet = predicted_image_monet[0]
    predicted_image_ukiyoe = predicted_image_ukiyoe[0]
    # Rescale the predicted image to match the input image's pixel range
    predicted_image_monet = ((predicted_image_monet + 1)*127.5).astype(np.uint8)
    predicted_image_ukiyoe = ((predicted_image_ukiyoe + 1)*127.5).astype(np.uint8)


    # The original image is converted from a numpy array to a format suitable for display.
    image = image[0].numpy()
    # Rescale the original image to match the input image's pixel range
    image = ((image + 1)*127.5).astype(np.uint8)
    

    # Create a figure with two subplots
    fig,axs = plt.subplots(1,3,figsize=(12,4))

    # Show the original input image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Ordinary Picture', fontsize=14)
    axs[0].axis('off')

    # Show the predicted output image in the second subplot
    axs[1].imshow(predicted_image_monet)
    axs[1].set_title('Monet Version', fontsize=14)
    axs[1].axis('off')

    axs[2].imshow(predicted_image_ukiyoe)
    axs[2].set_title('Ukiyoe Version', fontsize=14)
    axs[2].axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to a PIL image
    wandb_image = Image.open(buf)

    # Log the image to wandb
    wandb.log({f"epoch_{epoch}_translation": [wandb.Image(wandb_image)]})

    
    
    plt.show()
    plt.close(fig)




def training_loop(picture_train_dataset, paintings_training_dataset, picture_test_dataset, model, max_epochs, parent_directory, folder_name, continue_training=False, save_weights=True, patience=20):

    # Define the folder path to save the weights and losses
    folder_path = os.path.join(parent_directory, folder_name)

    # Create a new directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)


    if continue_training:
        # If training is being resumed, load the weights and losses from the requested epoch
        start_epoch = int(input("Enter the epoch you want to continue training from: "))

        # Calling the model with some sample input to initialize the variables
        dummy_input = (tf.zeros([1, 128, 128, 3]), tf.zeros([1, ]))
        _ = model(dummy_input,training=False)

        # Load the weights from the previous checkpoint
        model.load_weights(folder_path+f"/epoch_{start_epoch}.h5")

        # Load the saved losses
        with open(folder_path+'/epoch_losses.pickle', 'rb') as f:
            epoch_losses = pickle.load(f)

        print(f'\nContinuing training from Epoch {start_epoch}\n')

    else:
        # If starting fresh, initialize the start epoch
        start_epoch = 0

        # If starting fresh, initialize the losses dictionary

        epoch_losses = {
            "gen_a2b_loss": [],
            "gen_b2a_loss": [],
            "disc_a_loss": [],
            "disc_b_loss": [],
            "aux_class_a_loss": [],
            "aux_class_b_loss": [],
            "cycle_a_loss": [],
            "cycle_b_loss": [],
            "identity_a_loss": [],
            "identity_b_loss": [],
        }


    # Initialize the best loss and early stopping counter
    best_loss = float('inf')
    early_stopping_counter = 0


    # Combine the datasets
    dataset = tf.data.Dataset.zip((picture_train_dataset, paintings_training_dataset))

    # Iterating over each epoch
    for epoch in range(start_epoch,max_epochs):

        print(
                    f"\nEpoch {epoch+1}/{max_epochs}:\n"
                    "============\n"
                )

        # Dictionary to store losses for each batch in every epoch
        # a is photos, b is art
        batch_losses = {
            "gen_a2b_loss": [],
            "gen_b2a_loss": [],
            "disc_a_loss": [],
            "disc_b_loss": [],
            "aux_class_a_loss": [],
            "aux_class_b_loss": [],
            "cycle_a_loss": [],
            "cycle_b_loss": [],
            "identity_a_loss": [],
            "identity_b_loss": [],
        }

        # Iterating over each batch
        for step, (real_p, (real_a, real_a_labels)) in tqdm(enumerate(dataset), total=len(dataset)):
            # Train the model on the current batch
            fake_paintings, loss_dict = model.train_step((real_p, (real_a, real_a_labels)))

            # Append the batch losses to the batch_losses dictionary
            for loss_name, loss_value in loss_dict.items():
                batch_losses[loss_name].append(loss_value.numpy())

        # Periodically after every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Plot the fake painting images
            print(f"\nPlotting Images for Epoch {epoch+1}\n")
            plot_images(fake_paintings)

        # Periodically after every 10 epochs
        if (epoch + 1) % 10 == 0:

            # Plot data from test set
            print(f"\nPlotting images from Test Set\n")
            plot_single_translation(picture_test_dataset, model, epoch)

            # Save the model weights
            if save_weights:
                print(f"\nSaving weights for Epoch {epoch+1}\n")
                model.save_weights(folder_path+f'/epoch_{epoch+1}.h5')

            # Save the epoch_losses dictionary
            with open(folder_path+'/epoch_losses.pickle', 'wb') as f:
                pickle.dump(epoch_losses, f)

        # Append the mean batch losses at the end of each epoch to the epoch_losses dictionary
        for loss_name, loss_list in batch_losses.items():
            epoch_losses[loss_name].append(np.mean(loss_list))

        # Save weights if early stopping criteria is met
        if epoch_losses['gen_a2b_loss'][epoch] < best_loss:
            # Save the new best weights
            best_loss = epoch_losses['gen_a2b_loss'][epoch]
            early_stopping_counter = 0
            model.save_weights(folder_path+f'/best_generator_a2b_loss.h5')
        # Increment counter if not
        else:
            early_stopping_counter += 1


        # Print the losses periodically at the end of each epoch
        wandb.log({"Photo to Art Generator Loss": epoch_losses['gen_a2b_loss'][epoch],
                   "Art to Photo Generator Loss": epoch_losses['gen_b2a_loss'][epoch],
                  "Photo Discriminator Loss": epoch_losses['disc_a_loss'][epoch],
                  "Art Discriminator Loss":epoch_losses['disc_b_loss'][epoch],
                  "Photo Auxiliary Classifier Loss":epoch_losses['aux_class_a_loss'][epoch],
                  "Art Auxiliary Classifier Loss":epoch_losses['aux_class_b_loss'][epoch],
                  "Photo Cycle Loss":epoch_losses['cycle_a_loss'][epoch],
                  "Art Cycle Loss":epoch_losses['cycle_b_loss'][epoch],
                  "Photo Identity Loss":epoch_losses['identity_a_loss'][epoch],
                  "Art Identity Loss":epoch_losses['identity_b_loss'][epoch]}, step=epoch)
        print(
                    f"\nLosses:\n"
                    "-------\n"
                    f"Photo to Art Generator Loss\t=\t{epoch_losses['gen_a2b_loss'][epoch]:.4f}\n"
                    f"Art to Photo Generator Loss\t=\t{epoch_losses['gen_b2a_loss'][epoch]:.4f}\n"
                    f"Photo Discriminator Loss\t=\t{epoch_losses['disc_a_loss'][epoch]:.4f}\n"
                    f"Art Discriminator Loss\t\t=\t{epoch_losses['disc_b_loss'][epoch]:.4f}\n"
                    f"Photo Auxiliary Classifier Loss\t=\t{epoch_losses['aux_class_a_loss'][epoch]:.4f}\n"
                    f"Art Auxiliary Classifier Loss\t=\t{epoch_losses['aux_class_b_loss'][epoch]:.4f}\n"
                    f"Photo Cycle Loss\t\t=\t{epoch_losses['cycle_a_loss'][epoch]:.4f}\n"
                    f"Art Cycle Loss\t\t\t=\t{epoch_losses['cycle_b_loss'][epoch]:.4f}\n"
                    f"Photo Identity Loss\t\t=\t{epoch_losses['identity_a_loss'][epoch]:.4f}\n"
                    f"Art Identity Loss\t\t=\t{epoch_losses['identity_b_loss'][epoch]:.4f}\n"
                )
        print('-'*30)
        print('\n')

        # Break the loop if loss doesn't decrese for more than patient epochs
        if early_stopping_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    print("Training complete")
    
    
    
