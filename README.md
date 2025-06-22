# PRODIGY_GA_04
# Pix2Pix GAN — Image-to-Image Translation

This script demonstrates a basic **Pix2Pix GAN** implementation using TensorFlow and PyTorch (for input simulation). It supports both **Google Colab** and **local execution**, and is suitable for sketch-to-image tasks or any paired image translation use-case.

---

## 📌 Description
This project loads paired image data (input & target), preprocesses them for training, builds a UNet-based generator and a basic PatchGAN-style discriminator, and performs a sample training step using adversarial + L1 loss.

- ✅ Automatic handling of `.zip` uploads (both local and Colab-compatible)
- ✅ Converts input image to [-1, 1] range (as expected by Pix2Pix models)
- ✅ Custom UNet-style generator using downsampling and upsampling blocks
- ✅ Patch-based discriminator using concatenated input & target images
- ✅ L1 loss + GAN loss training logic

---

## 🔧 Required Packages
Install the following Python packages before running:

```bash
pip install tensorflow pillow numpy torch
```

For Google Colab users working with `diffusers`, also run:

```bash
pip install diffusers transformers accelerate safetensors
```

---

## 📁 How to Use

### 🔹 For Google Colab:
- Automatically triggers a file upload dialog
- Accepts `.zip` files containing paired training images
- Prompts for test image for normalization and feeding

### 🔹 For Local Execution:
- Asks for zip path using `input()`
- Asks for test image path using `input()`

---

## ✅ Output
- Prints generator summary
- Executes one training step
- Logs Generator and Discriminator loss

---

## 📂 Notes
- This implementation is minimal and meant for educational purposes
- You may expand training to full dataset loop, logging, and validation
- Generator architecture is UNet-like with skip connections

---

## 🧠 Credit
This script was built as part of an internship project exploring **Generative Adversarial Networks** using Pix2Pix framework. Useful for academic submissions and personal experiments!
