import os
import pathlib
import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

import tensorflow as tf
import pathlib

dataset_name = input("Enter dataset name (e.g., facades, maps, edges2shoes): ").strip()
_URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz"

path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True)

# Convert to Pathlib object
path_to_zip = pathlib.Path(path_to_zip)

# ✅ Actual extraction folder is:
# ~/.keras/datasets/{dataset_name}
PATH = path_to_zip.parent / dataset_name

print("✅ Dataset extracted to:", PATH)


# ====================== Load and preprocess images ======================
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    w = tf.shape(image)[1] // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

# Load sample training pairs
train_paths = tf.io.gfile.glob(str(PATH / 'train/*.jpg'))[:100]

# Convert to torch tensors for DGCAN training
input_images = []
target_images = []
for path in train_paths:
    inp, tgt = load_image_train(path)
    inp_resized = tf.image.resize(inp, [256, 256])
    tgt_resized = tf.image.resize(tgt, [256, 256])

    inp_np = inp_resized.numpy()
    tgt_np = tgt_resized.numpy()

    input_images.append(torch.tensor(inp_np.transpose(2, 0, 1)))
    target_images.append(torch.tensor(tgt_np.transpose(2, 0, 1)))

input_images = torch.stack(input_images)
target_images = torch.stack(target_images)

print("✅ Loaded and preprocessed CMP Facades samples")

# ====================== U-Net Generator & Discriminator ======================
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=not apply_batchnorm))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def build_generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2,
                                           padding='same',
                                           kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                           activation='tanh')
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def build_discriminator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    targets = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.Concatenate()([inputs, targets])
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=[inputs, targets], outputs=x)

# ====================== Loss and Training ======================
generator = build_generator()
discriminator = build_discriminator()
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    adv_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return adv_loss + (LAMBDA * l1_loss)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_obj(tf.ones_like(real_output), real_output)
    fake_loss = loss_obj(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_fake_output = discriminator([input_image, gen_output], training=True)
        gen_loss = generator_loss(disc_fake_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    return gen_loss, disc_loss, gen_output

def show_image(tensor_img):
    img = (tensor_img[0].numpy() + 1) / 2.0
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# ====================== Training Loop ======================
EPOCHS = 5
STEPS = 1000
for step in range(STEPS):
    i = step % len(input_images)
    sample_input = tf.convert_to_tensor(input_images[i].numpy().transpose(1, 2, 0))[tf.newaxis, ...]
    sample_target = tf.convert_to_tensor(target_images[i].numpy().transpose(1, 2, 0))[tf.newaxis, ...]
    g_loss, d_loss, g_output = train_step(sample_input, sample_target)

    if (step + 1) % 1000 == 0:
        print(f"\n✅ Step {step+1}/{STEPS}")
        print("Generator Loss:", g_loss.numpy())
        print("Discriminator Loss:", d_loss.numpy())
        print("🖼️ Generated sample:")
        show_image(g_output)

print("✅ Training complete")
