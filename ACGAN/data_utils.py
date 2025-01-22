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
from model_opt import *
from model_arch import *
from model_eval import *
from PIL import Image
import wandb
import io
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = (128, 128)
BUFFER_SIZE = 10000
BATCH_SIZE = 32

def load_style_datasets(style_name, training_split, testing_split, storage_directory='Data'):

    training_dataset = tfds.load(f"cycle_gan/{style_name}", split=training_split, as_supervised=True, shuffle_files=False,
                         data_dir=storage_directory, download=True)
    testing_dataset = tfds.load(f"cycle_gan/{style_name}", split=testing_split, as_supervised=True, shuffle_files=False,
                        data_dir=storage_directory, download=True)
    return training_dataset, testing_dataset

# Function to add labels to dataset
def add_labels_to_dataset(dataset, label):
    return dataset.map(lambda x, _: (x, label))

# Function to remove labels from dataset (if any)
def remove_labels_from_dataset(dataset):
    return dataset.map(lambda image, _: image)

# Function to calculate and print dataset sizes
def print_dataset_sizes(datasets, dataset_descriptions):
    for dataset, description in zip(datasets, dataset_descriptions):
        size = tf.data.experimental.cardinality(dataset).numpy()
        print(f'Size of {description}:\t{size}')
        
def combine_and_adjust_datasets(dataset_one, dataset_two):
    # Determine the smaller size between the two datasets
    size_one = tf.data.experimental.cardinality(dataset_one).numpy()
    size_two = tf.data.experimental.cardinality(dataset_two).numpy()
    smaller_size = min(size_one, size_two)
    
    # Adjust both datasets to be of the same size
    adjusted_dataset_one = dataset_one.take(smaller_size)
    adjusted_dataset_two = dataset_two.take(smaller_size)
    
    # Splitting the datasets into training and testing sets (80% training, 20% testing)
    train_size = int(smaller_size * 0.8)
    
    train_dataset_one = adjusted_dataset_one.take(train_size)
    test_dataset_one = adjusted_dataset_one.skip(train_size)
    
    train_dataset_two = adjusted_dataset_two.take(train_size)
    test_dataset_two = adjusted_dataset_two.skip(train_size)
    
    return train_dataset_one, test_dataset_one, train_dataset_two, test_dataset_two


def preprocess_image(image, label=None):
    # Normalize the pixel values to range from -1 to 1
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0

    # Resize the image
    image = tf.image.resize(image, IMG_SIZE)

    return (image, label) if label is not None else image

def prepare_dataset(dataset):
    # Apply preprocessing
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

    
