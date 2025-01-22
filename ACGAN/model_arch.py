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
from model_eval import *


class LabelAlphaBlending(layers.Layer):
    """
    Layer that blends label embeddings for style transfer using alpha blending coefficients.
    """

    def __init__(self, **kwargs):
        super(LabelAlphaBlending, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_style_a = layers.Embedding(1, 50)
        self.embedding_style_b = layers.Embedding(1, 50)

        img_height, img_width = input_shape[0][1], input_shape[0][2]
        self.dense_style_a = layers.Dense(img_height * img_width)
        self.dense_style_b = layers.Dense(img_height * img_width)

    def call(self, inputs):
        input_image, input_label = inputs

        alpha = tf.reshape(input_label, (-1, 1))
        alpha_complement = tf.reshape(1 - input_label, (-1, 1))

        label_embedding_style_a = self.embedding_style_a(tf.zeros_like(alpha))
        label_embedding_style_b = self.embedding_style_b(tf.zeros_like(alpha_complement))

        label_embedding_style_a = self.dense_style_a(label_embedding_style_a)
        label_embedding_style_b = self.dense_style_b(label_embedding_style_b)

        label_embedding = alpha * label_embedding_style_a + alpha_complement * label_embedding_style_b
        label_embedding = tf.keras.layers.GlobalAveragePooling1D()(label_embedding)
        label_embedding = layers.Reshape((input_image.shape[1], input_image.shape[2], 1))(label_embedding)

        x = tf.concat([input_image, label_embedding], axis=-1)

        return x
    
    
def residual_block(input_tensor, input_label, filters, block_idx, concat_label=True):
    """
    Residual block for the generator network.
    """
    init_weights = initializers.RandomNormal(stddev=0.02)

    if (block_idx + 1) % 3 == 0 and concat_label:
        alpha_concat = LabelAlphaBlending(name=f'ResidualBlock_{block_idx+1}_LabelAlphaBlending')
        x = alpha_concat([input_tensor, input_label])
    else:
        x = input_tensor

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer=init_weights, name=f'ResidualBlock_{block_idx+1}_Conv2D_1')(x)
    x = InstanceNormalization(axis=-1, name=f'ResidualBlock_{block_idx+1}_InstanceNorm_1')(x)
    x = layers.Activation('relu', name=f'ResidualBlock_{block_idx+1}_ReLU_1')(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer=init_weights, name=f'ResidualBlock_{block_idx+1}_Conv2D_2')(x)
    x = InstanceNormalization(axis=-1, name=f'ResidualBlock_{block_idx+1}_InstanceNorm_2')(x)

    x = layers.Add(name=f'ResidualBlock_{block_idx+1}_Addition')([x, input_tensor])

    return x

def build_generator(input_shape, name, label_shape=(1,), num_styles=2, num_residual_blocks=9, filters=64, concat_labels_resnet=True, concat_labels_upsample=True):
    """
    Builds the generator network for the CycleGAN model.
    """
    init_weights = initializers.RandomNormal(stddev=0.02)

    input_image = layers.Input(shape=input_shape, name='ImageInput')
    input_label = layers.Input(shape=label_shape, name='LabelInput', dtype=tf.float32)

    x = LabelAlphaBlending(name='LabelAlphaBlending')([input_image, input_label])

    x = layers.Conv2D(filters, kernel_size=7, strides=1, padding='same', kernel_initializer=init_weights, name='Conv2D_1')(x)
    x = InstanceNormalization(axis=-1, name='InstanceNorm_1')(x)
    x = layers.Activation('relu', name='ReLU_1')(x)

    for i in range(2):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=init_weights, name=f'Conv2D_Encoder_{i+1}')(x)
        x = InstanceNormalization(axis=-1, name=f'InstanceNorm_Encoder_{i+1}')(x)
        x = layers.Activation('relu', name=f'ReLU_Encoder_{i+1}')(x)

    for i in range(num_residual_blocks):
        x = residual_block(x, input_label, filters, i, concat_labels_resnet)

    for i in range(2):
        filters //= 2
        x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=init_weights, name=f'Conv2DTranspose_Decoder_{i+1}')(x)
        x = InstanceNormalization(axis=-1, name=f'InstanceNorm_Decoder_{i+1}')(x)
        x = layers.Activation('relu', name=f'ReLU_Decoder_{i+1}')(x)
        if i == 0 and concat_labels_upsample:
            x = LabelAlphaBlending(name='LabelAlphaBlending_Upsampling')([x, input_label])

    x = layers.Conv2DTranspose(3, kernel_size=7, strides=1, padding='same', kernel_initializer=init_weights, name='Conv2DTranspose_Output')(x)
    x = InstanceNormalization(axis=-1, name='InstanceNorm_Output')(x)
    output_tensor = layers.Activation('tanh', name='TanH_Output')(x)

    return tf.keras.Model(inputs=[input_image, input_label], outputs=output_tensor, name=name)


def build_discriminator(input_shape, name, label_shape=(1,), num_styles=2, filters=64):
    """
    Builds the discriminator network for the CycleGAN model.
    """
    init_weights = initializers.RandomNormal(stddev=0.02)

    input_image = layers.Input(shape=input_shape, name='ImageInput')
    input_label = layers.Input(shape=label_shape, name='LabelInput')

    label_embedding = layers.Embedding(num_styles, input_shape[0] * input_shape[1], name='LabelEmbedding')(input_label)
    label_embedding = layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1), name='LabelReshape')(label_embedding)
    x = layers.Concatenate(axis=-1, name='Concatenate')([input_image, label_embedding])

    x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same', kernel_initializer=init_weights, name='Conv2D_1')(x)
    x = layers.LeakyReLU(alpha=0.2, name='LeakyReLU_1')(x)

    for i in range(3):
        filters *= 2
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same', kernel_initializer=init_weights, name=f'Conv2D_{i+2}')(x)
        x = InstanceNormalization(axis=-1, name=f'InstanceNorm_{i+1}')(x)
        x = layers.LeakyReLU(alpha=0.2, name=f'LeakyReLU_{i+2}')(x)

    x = layers.Conv2D(filters, kernel_size=4, strides=1, padding='same', kernel_initializer=init_weights, name=f'Conv2D_{i+1+2}')(x)
    x = InstanceNormalization(axis=-1, name=f'InstanceNorm_{i+1+1}')(x)
    x = layers.LeakyReLU(alpha=0.2, name=f'LeakyReLU_{i+1+2}')(x)
    patch_output = layers.Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=init_weights, name='Patch_Output')(x)

    class_output = layers.GlobalAveragePooling2D(name='GlobalAveragePooling')(x)
    class_output = layers.Dense(num_styles, activation='softmax', name='ClassOutput')(class_output)

    return tf.keras.Model(inputs=[input_image, input_label], outputs=[patch_output, class_output], name=name)

class CycleGANWithAuxClassifier(tf.keras.Model):
    """
    A Keras model implementing the Auxiliary Classifier CycleGAN architecture.
    """

    def __init__(self, generator_a2b, generator_b2a, discriminator_a, discriminator_b, num_styles, name='CycleGAN_with_AuxClassifier', lambda_cycle=10.0, lambda_identity=0.5, lambda_aux=1):
        super(CycleGANWithAuxClassifier, self).__init__(name=name)
        self.generator_a2b = generator_a2b
        self.generator_b2a = generator_b2a
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b
        self.num_styles = num_styles
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_aux = lambda_aux

    def compile(self, gen_a2b_optimizer, gen_b2a_optimizer, disc_a_optimizer, disc_b_optimizer, gen_loss_fn, disc_loss_fn, aux_class_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(CycleGANWithAuxClassifier, self).compile()
        self.gen_a2b_optimizer = gen_a2b_optimizer
        self.gen_b2a_optimizer = gen_b2a_optimizer
        self.disc_a_optimizer = disc_a_optimizer
        self.disc_b_optimizer = disc_b_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.aux_class_loss_fn = aux_class_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    @tf.function
    def train_step(self, data):
        real_a, (real_b, real_b_labels) = data

        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
            fake_b = self.generator_a2b([real_a, real_b_labels], training=True)
            fake_a = self.generator_b2a([real_b, real_b_labels], training=True)

            cycled_a = self.generator_b2a([fake_b, real_b_labels], training=True)
            cycled_b = self.generator_a2b([fake_a, real_b_labels], training=True)

            same_a = self.generator_b2a([real_a, real_b_labels], training=True)
            same_b = self.generator_a2b([real_b, real_b_labels], training=True)

            disc_real_a, aux_class_real_a = self.discriminator_a([real_a, real_b_labels], training=False)
            disc_real_b, aux_class_real_b = self.discriminator_b([real_b, real_b_labels], training=False)
            disc_fake_a, aux_class_fake_a = self.discriminator_a([fake_a, real_b_labels], training=False)
            disc_fake_b, aux_class_fake_b = self.discriminator_b([fake_b, real_b_labels], training=False)

            gen_a2b_loss = self.gen_loss_fn(disc_fake_b)
            gen_b2a_loss = self.gen_loss_fn(disc_fake_a)

            disc_real_a, aux_class_real_a = self.discriminator_a([real_a, real_b_labels], training=True)
            disc_real_b, aux_class_real_b = self.discriminator_b([real_b, real_b_labels], training=True)
            disc_fake_a, aux_class_fake_a = self.discriminator_a([fake_a, real_b_labels], training=True)
            disc_fake_b, aux_class_fake_b = self.discriminator_b([fake_b, real_b_labels], training=True)

            disc_a_loss = self.disc_loss_fn(disc_real_a, disc_fake_a)
            disc_b_loss = self.disc_loss_fn(disc_real_b, disc_fake_b)

            aux_class_real_a_loss = self.aux_class_loss_fn(real_b_labels, aux_class_real_a) * self.lambda_aux
            aux_class_fake_a_loss = self.aux_class_loss_fn(real_b_labels, aux_class_fake_a) * self.lambda_aux
            aux_class_real_b_loss = self.aux_class_loss_fn(real_b_labels, aux_class_real_b) * self.lambda_aux
            aux_class_fake_b_loss = self.aux_class_loss_fn(real_b_labels, aux_class_fake_b) * self.lambda_aux

            cycle_a_loss = self.cycle_loss_fn(real_a, cycled_a) * self.lambda_cycle
            cycle_b_loss = self.cycle_loss_fn(real_b, cycled_b) * self.lambda_cycle

            identity_a_loss = self.identity_loss_fn(real_a, same_a) * self.lambda_identity
            identity_b_loss = self.identity_loss_fn(real_b, same_b) * self.lambda_identity

            total_gen_a2b_loss = gen_a2b_loss + cycle_b_loss + cycle_a_loss + identity_b_loss + aux_class_fake_b_loss
            total_gen_b2a_loss = gen_b2a_loss + cycle_b_loss + cycle_a_loss + identity_a_loss + aux_class_fake_a_loss
            total_disc_a_loss = disc_a_loss + aux_class_real_a_loss + aux_class_fake_a_loss
            total_disc_b_loss = disc_b_loss + aux_class_real_b_loss + aux_class_fake_b_loss

        gen_a2b_gradients = gen_tape.gradient(total_gen_a2b_loss, self.generator_a2b.trainable_variables)
        gen_b2a_gradients = gen_tape.gradient(total_gen_b2a_loss, self.generator_b2a.trainable_variables)
        disc_a_gradients = disc_tape.gradient(total_disc_a_loss, self.discriminator_a.trainable_variables)
        disc_b_gradients = disc_tape.gradient(total_disc_b_loss, self.discriminator_b.trainable_variables)

        self.gen_a2b_optimizer.apply_gradients(zip(gen_a2b_gradients, self.generator_a2b.trainable_variables))
        self.gen_b2a_optimizer.apply_gradients(zip(gen_b2a_gradients, self.generator_b2a.trainable_variables))
        self.disc_a_optimizer.apply_gradients(zip(disc_a_gradients, self.discriminator_a.trainable_variables))
        self.disc_b_optimizer.apply_gradients(zip(disc_b_gradients, self.discriminator_b.trainable_variables))

        return (fake_b, real_b_labels), {
            "gen_a2b_loss": total_gen_a2b_loss,
            "gen_b2a_loss": total_gen_b2a_loss,
            "disc_a_loss": total_disc_a_loss,
            "disc_b_loss": total_disc_b_loss,
            "aux_class_a_loss": aux_class_real_a_loss + aux_class_fake_a_loss,
            "aux_class_b_loss": aux_class_real_b_loss + aux_class_fake_b_loss,
            "cycle_a_loss": cycle_a_loss,
            "cycle_b_loss": cycle_b_loss,
            "identity_a_loss": identity_a_loss,
            "identity_b_loss": identity_b_loss,
        }

    def call(self, inputs, training=None):
        input_image, input_label = inputs
        fake_image = self.generator_a2b([input_image, input_label], training=training)
        return fake_image