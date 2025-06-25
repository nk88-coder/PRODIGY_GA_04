# PRODIGY_GA_04
# Pix2Pix GAN — Image-to-Image Translation

🔥 DGCAN Pix2Pix GAN Trainer (Facades / Maps / Edges2Shoes)

This project trains a Pix2Pix-style GAN using:

- TensorFlow for U-Net generator & discriminator
- PyTorch for preprocessing (converts paired image tensors)
- Real paired datasets like facades, maps, edges2shoes from Efros's official Pix2Pix site

---

📦 Features:

✅ Downloads & extracts the dataset automatically  
✅ Splits & normalizes paired images (input | target)  
✅ Resizes to 256x256  
✅ Trains U-Net Generator + PatchGAN Discriminator  
✅ Displays generated output every 1000 steps  
✅ Easy to read, modify, and run

---

🚀 How to Run:

1. Install required libraries:
   pip install tensorflow torch matplotlib pillow

2. Run the training script:
   python train.py

3. When prompted, enter:
   facades
   or
   maps
   or
   edges2shoes

---

📁 Dataset Path:

Automatically downloaded from:
http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz

Extracted to:
~/.keras/datasets/{dataset_name}_extracted/{dataset_name}/train/

Each training image is a side-by-side (input | target) image.

---

🧠 Model Details:

Generator (U-Net):
- 8 downsampling layers (Conv2D → LeakyReLU)
- 7 upsampling layers (Conv2DTranspose → ReLU)
- Skip connections from encoder to decoder
- Final output with tanh activation

Discriminator (PatchGAN):
- Input: concatenated input + target
- Layers: Conv2D → LeakyReLU → Flatten → Dense(1)
- Outputs a real/fake score

---

📊 Training Config:

- Loss Function:
  Generator → BinaryCrossentropy + L1 Loss  
  Discriminator → BinaryCrossentropy  
- Optimizer: Adam (lr = 2e-4, beta1 = 0.5)
- Trains for 1000 steps
- Shows output image every 1000 steps

---

🖼️ Sample Visualization:

Images are denormalized and shown using matplotlib:
(tensor + 1) / 2.0 → [0, 1] range

---

⚙️ Customization Tips:

- Increase STEPS if needed
- Wrap it inside EPOCHS if you want full-loop training
- Add your own dataset (just match the paired .jpg format)

---

🧼 Errors?

❌ No valid images loaded  
→ Check if extracted path is correct and has train/*.jpg files

❌ Shape mismatch  
→ Ensure images are resized to 256x256 before stacking

---

💼 Notes:

- Works with both CPU and GPU
- Great for demonstrating GANs in internships or demo projects
- Uses minimal dependencies and keeps code readable

---

## 🧠 Credit

This script was built as part of an internship project exploring **Generative Adversarial Networks** using the Pix2Pix framework. Useful for academic submissions and personal experiments!
