import os
import zipfile
import numpy as np
from PIL import Image
import tensorflow as tf
import torch

# ====================== ZIP UPLOAD ======================
def extract_zip(zip_path, extract_to="/content/pix2pix_data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")

# Colab or local file handling
try:
    from google.colab import files
    uploaded = files.upload()
    for filename in uploaded.keys():
        extract_zip(filename)
except ImportError:
    zip_path = input("Enter path to your local zip file: ")
    extract_zip(zip_path)

# ====================== MOCK TENSORS FOR TEST ======================
input_images = torch.randn(4, 3, 256, 256)   # Simulated input
target_images = torch.randn(4, 3, 256, 256)  # Simulated ground truth

# ====================== IMAGE LOADER ======================
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    image_np = np.array(image).astype(np.float32)
    image_np = (image_np / 127.5) - 1.0
    return np.expand_dims(image_np, axis=0)  # shape: (1, 256, 256, 3)

image_path = input("Enter image path: ")
input_image = load_and_preprocess_image(image_path)

# ====================== UNET GENERATOR ======================
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

generator = build_generator()
generator.summary()

# ====================== LOSS & TRAINING SETUP ======================
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

# Mock discriminator
inputs = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
targets = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
combined = tf.keras.layers.Concatenate()([inputs, targets])
d = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(combined)
d = tf.keras.layers.LeakyReLU()(d)
d = tf.keras.layers.Flatten()(d)
d = tf.keras.layers.Dense(1)(d)
discriminator = tf.keras.Model(inputs=[inputs, targets], outputs=d)

# ====================== TRAIN STEP ======================
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

    return gen_loss, disc_loss

# ========== CALL TRAIN STEP (EXAMPLE) ==========
train_gen_loss, train_disc_loss = train_step(tf.convert_to_tensor(input_image), tf.convert_to_tensor(input_image))
print("✅ Train step complete")
print("Generator Loss:", train_gen_loss.numpy())
print("Discriminator Loss:", train_disc_loss.numpy())
