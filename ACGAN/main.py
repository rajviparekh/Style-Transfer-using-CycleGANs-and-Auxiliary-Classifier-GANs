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
from model_eval import *
import wandb

wandb.init(project="CV_project", entity="aryamohan23")



# Load datasets for Monet paintings
monet_datasets_style = "monet2photo"
monet_training_dataset, monet_testing_dataset = load_style_datasets(monet_datasets_style, "trainA", "testA")

# Load datasets for Ukiyo-e art
ukiyoe_datasets_style = "ukiyoe2photo"
ukiyoe_training_dataset, ukiyoe_testing_dataset = load_style_datasets(ukiyoe_datasets_style, "trainA", "testA")

# Load datasets for ordinary photographs (aligned with Ukiyo-e for contrast)
photographs_datasets_style = "ukiyoe2photo"
ordinary_photos_training_dataset, ordinary_photos_testing_dataset = load_style_datasets(photographs_datasets_style, "trainB", "testB")

# Label definitions
label_monet = tf.constant(0, dtype=tf.int32)
label_ukiyoe = tf.constant(1, dtype=tf.int32)

# Adding labels to datasets
monet_training_dataset = add_labels_to_dataset(monet_training_dataset, label_monet)
monet_testing_dataset = add_labels_to_dataset(monet_testing_dataset, label_monet)

ukiyoe_training_dataset = add_labels_to_dataset(ukiyoe_training_dataset, label_ukiyoe)
ukiyoe_testing_dataset = add_labels_to_dataset(ukiyoe_testing_dataset, label_ukiyoe)

# Removing labels from ordinary photos datasets
ordinary_photos_training_dataset = remove_labels_from_dataset(ordinary_photos_training_dataset)
ordinary_photos_testing_dataset = remove_labels_from_dataset(ordinary_photos_testing_dataset)


# Combine the Monet and Ukiyo-e datasets for both training and testing
combined_monet_dataset = monet_training_dataset.concatenate(monet_testing_dataset)
combined_ukiyoe_dataset = ukiyoe_training_dataset.concatenate(ukiyoe_testing_dataset)


# Applying the adjusted combine and split function
monet_train_ds, monet_test_ds, ukiyoe_train_ds, ukiyoe_test_ds = combine_and_adjust_datasets(combined_monet_dataset, combined_ukiyoe_dataset)

# Calculating and reporting dataset sizes
datasets = [monet_train_ds, monet_test_ds, ukiyoe_train_ds, ukiyoe_test_ds]
descriptions = ['Monet training', 'Monet testing', 'Ukiyo-e training', 'Ukiyo-e testing']
print_dataset_sizes(datasets, descriptions)

# Combining training and testing datasets of both styles
paintings_training_dataset = monet_train_ds.concatenate(ukiyoe_train_ds)
paintings_testing_dataset = monet_test_ds.concatenate(ukiyoe_test_ds)

# Reporting the final combined dataset sizes
final_datasets = [paintings_training_dataset, paintings_testing_dataset]
final_descriptions = ['combined training', 'combined testing']
print_dataset_sizes(final_datasets, final_descriptions)

# Preprocessing and preparing the Paintings dataset
paintings_training_dataset = prepare_dataset(paintings_training_dataset)
paintings_testing_dataset = prepare_dataset(paintings_testing_dataset)

# Preprocessing and preparing the Pictures dataset
picture_train_dataset = prepare_dataset(ordinary_photos_training_dataset)
picture_test_dataset = prepare_dataset(ordinary_photos_testing_dataset)

# plot_pictures = next(iter(picture_train_dataset))
# plot_images(plot_pictures, num_images=20)

# plot_paintings = next(iter(paintings_training_dataset))
# plot_images(plot_paintings, num_images=20)



# Adam optimizer for the generator A2B with learning rate decay
gen_a2b_optimizer = tf.keras.optimizers.Adam(learning_rate=create_lr_schedule(0.00075, 100, paintings_training_dataset, decay_rate=0.85), beta_1=0.9)

# Adam optimizer for the generator B2A with learning rate decay
gen_b2a_optimizer = tf.keras.optimizers.Adam(learning_rate=create_lr_schedule(0.00075, 100, paintings_training_dataset, decay_rate=0.85), beta_1=0.9)

# Adam optimizer for the discriminator A with learning rate decay
disc_a_optimizer = tf.keras.optimizers.Adam(learning_rate=create_lr_schedule(1e-5, 100, paintings_training_dataset, decay_rate=0.85), beta_1=0.1)

# Adam optimizer for the discriminator B with learning rate decay
disc_b_optimizer = tf.keras.optimizers.Adam(learning_rate=create_lr_schedule(1e-5, 100, paintings_training_dataset, decay_rate=0.85), beta_1=0.1)

# Define the model's input shape and the number of styles
input_shape = (128, 128, 3)
label_shape = (1,)
num_styles = 2

# Build the generator and discriminator models
generator_a2b = build_generator(input_shape, name='Generator_A2B', label_shape=label_shape, num_styles=num_styles)
generator_b2a = build_generator(input_shape, name='Generator_B2A', label_shape=label_shape, num_styles=num_styles)
discriminator_a = build_discriminator(input_shape, name='Discriminator_A', label_shape=label_shape, num_styles=num_styles)
discriminator_b = build_discriminator(input_shape, name='Discriminator_B', label_shape=label_shape, num_styles=num_styles)

# Create the CycleGAN model with the auxiliary classifier
model = CycleGANWithAuxClassifier(generator_a2b, generator_b2a, discriminator_a, discriminator_b, num_styles, lambda_aux=2)

# Compile the model with the defined loss functions and optimizers
model.compile(gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer, generator_loss, discriminator_loss, aux_class_loss, cycle_loss, identity_loss,)

    
parent_directory = "ModelWeights/"
folder_name = "Final_Weights"


training_loop(picture_train_dataset, paintings_training_dataset, picture_test_dataset, model, max_epochs = 20, parent_directory = parent_directory, folder_name = folder_name, continue_training = False, save_weights = True)


epoch_losses, model = load_model(parent_directory, folder_name, generator_a2b, generator_b2a, discriminator_a, discriminator_b, num_styles, gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer, generator_loss, discriminator_loss, aux_class_loss, cycle_loss, identity_loss)


plot_losses(epoch_losses)

# Plotting the images from the test data
plot_generated_images(model, picture_test_dataset)