# Gaussian Color Image Denoising (Restormer + DRUNet)
This project implements a two-stage cascade model for effective Gaussian color image denoising using PyTorch.
# Project Description
The core idea is to combine the strengths of two different network architectures:
  1.Stage 1: Pre-filtering (Restormer) A pre-trained Restormer model is used for an initial, powerful denoising pass on the noisy input image.
  2.Stage 2: Refinement (DRUNet) A DRUNet (U-Net based architecture) is then trained to refine the output from the first stage. It takes a 6-channel input (concatenation of the original noisy image and the pre-        filtered image) and learns to predict the final, clean image.
This cascade approach leverages the Restormer's ability to handle heavy noise and the DRUNet's effectiveness in refining details and removing residual artifacts.  
# Dataset
This project is trained and evaluated on the BSDS300 dataset. The code includes utilities for downloading and preparing this dataset for training and validation.
# Features
  Implementation of a two-stage denoising pipeline.

  Integration of a pre-trained Restormer model for initial denoising.

  Training and evaluation scripts for the DRUNet refinement network.

  Calculation of evaluation metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
 # How to Use
 # The denoising.ipynb notebook provides a step-by-step walkthrough:

  Setup: Install necessary dependencies.

  Download Dependencies: Download the pre-trained Restormer model checkpoint.

  Prepare Data: Download and prepare the BSDS300 dataset, creating noisy-clean image pairs.

  Stage 1 (Pre-filtering): Run the Restormer model on the noisy dataset images and save the pre-filtered results.

  Stage 2 (Training): Train the DRUNet model using the 6-channel input (noisy image + pre-filtered image) to predict the clean image.

  Evaluation: Evaluate the trained model on the validation set and print the final PSNR and SSIM scores.
# Requirements
# The project relies on the following major libraries:

  torch (PyTorch)

  torchvision

  numpy

  opencv-python (cv2)

  matplotlib
  
  scikit-image (skimage)  
