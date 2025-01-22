# Neural Style Transfer with Multi-Style AC-GAN

## Overview
This project explores **Neural Style Transfer (NST)** using an **Auxiliary Classifier Cycle GAN (AC-CycleGAN)** to apply and blend multiple artistic styles onto images. We implement NST to enable the transformation of images into artistic renditions of styles such as **Monet** and **Ukiyo-e**, offering a flexible and scalable approach to style transfer.

## Objectives
1. **Single Style Transfer:** Implement NST using CycleGAN to apply one artistic style to an image.
2. **Multi-Style Transfer:** Extend the model using AC-CycleGAN to learn and apply multiple styles.
3. **Style Blending:** Generate output images that combine multiple learned styles for creative synthesis.

## Methodology
Our approach leverages AC-CycleGAN, which enhances the traditional CycleGAN framework by introducing an auxiliary classifier for multi-style learning. The model consists of:
- **Generator:** Translates images from one style domain to another.
- **Discriminator:** Distinguishes real from generated images and classifies styles.
- **Loss Functions:** Adversarial loss ensures style realism, while cycle consistency loss preserves content.

### Model Architecture
1. **CycleGAN Framework:** Enables unpaired style transfer across domains.
2. **PatchGAN Discriminator:** Focuses on texture and fine details in images.
3. **Residual Blocks:** Preserve content features while applying artistic styles.

## Dataset
We used two primary datasets for training:
- **Monet Paintings:** 1,200 images from the TensorFlow Dataset (monet2photo).
- **Ukiyo-e Paintings:** 1,000 images from the TensorFlow Dataset (ukiyoe2photo).
- **Regular Landscape Images:** 6,000 images for content representation.

## Training Process
- The data was preprocessed and normalized.
- The model was trained using adversarial and cycle consistency losses.
- Performance was evaluated using **Fréchet Inception Distance (FID)** to measure the quality of generated images.

## Results
- The baseline CycleGAN models achieved an FID of **0.29 (Monet)** and **0.43 (Ukiyo-e)**.
- Our AC-CycleGAN model achieved an FID of **1.87 (Monet)** and **4.62 (Ukiyo-e)**, showing promising style blending capabilities.
- The model successfully applied single and blended artistic styles to input images.

## Challenges Encountered
- **Checkerboard Artifacts:** Addressed by tuning architectural configurations and exploring upsampling strategies.
- **Style Generalization:** Further improvements needed for better style blending fidelity.

## Future Improvements
- Implementing attention mechanisms for enhanced style blending.
- Exploring larger and more diverse datasets to improve generalization.
- Addressing artifacts using sub-pixel convolution or improved upsampling techniques.



---

