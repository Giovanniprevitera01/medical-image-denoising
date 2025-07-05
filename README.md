# 🧠 Medical Imaging Denoising


A medical image denoising project focused on pancreatic tumor CT scans. It compares deep learning (Autoencoder) and classical machine learning (Ridge Regression + Deconvolution) approaches for noise reduction.

The goal: enhance image quality, improve diagnostic accuracy, and support AI models in medical imaging.

## 📂 Project Structure
''' bash
Medical imaging denoising/
│
├── Autoencoder/
│   ├── Dataset/
│   │   ├── train/
│   │   └── test/
│   ├── model.py
│   └── denoising1.ipynb
│
├── Regressione-Deconvoluzione/
│   ├── Dataset/
│   │   ├── train/
│   │   └── test/
│   └── denoising2.ipynb
│
├── assets/
│   ├── architecture.png
│   ├── workflow.png
│   ├── original_sample.png
│   ├── noisy_sample.png
│   ├── autoencoder_result.png
│   ├── regression_result.png
│   ├── metrics_comparison.png
│
├── README.md
└── LICENSE
'''

## 🖋️ Overview

This project implements and compares two denoising strategies for medical images:

    🧠 Deep Learning Autoencoder: a convolutional neural network trained to reconstruct clean images from noisy inputs.

    📈 Ridge Regression + Deconvolution: classical techniques enhanced with median filtering.

Both methods are evaluated on pancreatic tumor CT scans with artificial Gaussian noise.

## 🚀 Workflow

    Dataset Preparation: Train/Test split of pancreatic tumor CT images.

    Noise Injection: Add Gaussian noise (noise_factor=0.3).

    Denoising: Apply Autoencoder or classical ML techniques.

    Evaluation: Compute PSNR, SSIM, and MSE metrics.

## 🧠 Autoencoder (Deep Learning)
Layer Type	Filters	Kernel Size	Strides	Activation	Padding
Conv2D	15	5x5	1	ReLU	Same
Conv2DTranspose	15	5x5	1	ReLU	Same

### 📂 Files: Autoencoder/model.py, Autoencoder/denoising1.ipynb

## 📈 Ridge Regression + Deconvolution (Classical ML)
Step	Description
Ridge Regression	Predict clean pixels (α=0.1)
Median Filtering	Post-processing for smoothing
FFT Deconvolution	Remove blur and residual noise

### 📂 File: Regressione-Deconvoluzione/denoising2.ipynb

## 📊 Results
Original	Noisy	Autoencoder	Ridge + Deconvolution
	
	
	
## 📈 Metric Comparison

Technique	PSNR ↑	SSIM ↑	MSE ↓
Noisy Input	18.4 dB	0.65	0.012
CNN Autoencoder	28.2 dB	0.92	0.002
Ridge + Median Filter	24.1 dB	0.85	0.006
Deconvolution + Median Filter	26.3 dB	0.88	0.004

## 📁 Dataset

Autoencoder/Dataset/train/
Autoencoder/Dataset/test/
Regressione-Deconvoluzione/Dataset/train/
Regressione-Deconvoluzione/Dataset/test/

## 💻 Getting Started
1️⃣ Clone the repository
'''bash
git clone https://github.com/<your-username>/medical-imaging-denoising.git
cd medical-imaging-denoising

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run the notebooks

cd Autoencoder
jupyter notebook denoising1.ipynb

cd ../Regressione-Deconvoluzione
jupyter notebook denoising2.ipynb
'''

## 📜 License

This project is licensed under the MIT License - see LICENSE.

## 👨‍💻 Author

Giovanni Previtera
📧 Email
🌐 GitHub🧠 Medical Imaging Denoising


