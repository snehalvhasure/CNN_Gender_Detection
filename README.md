## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![NumPy](https://img.shields.io/badge/NumPy-DataProcessing-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)

---

# 🧠 CNN Gender Detection using Deep Learning

This project is a Convolutional Neural Network (CNN) based image classification system that detects gender (Male/Female) from facial images.

---

## 🚀 Project Overview

The model is trained using a custom dataset of male and female face images. It uses a CNN architecture built with TensorFlow/Keras to classify images into two categories:

- Female → 0
- Male → 1

---

## 📊 Dataset

- Custom dataset of facial images
- Images are resized to **100 x 150**
- Labels:
  - Female → 0  
  - Male → 1  

Dataset is loaded using OpenCV and processed into NumPy arrays.

---

## 🧠 Model Architecture

The CNN model consists of:

- Input Layer (150, 100, 3)
- Conv2D + ReLU
- MaxPooling2D
- Conv2D + ReLU
- MaxPooling2D
- Conv2D + ReLU
- MaxPooling2D
- Flatten Layer
- Dense (512 neurons)
- Dense (1024 neurons)
- Dense (512 neurons)
- Output Layer (Softmax - 2 classes)

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 🏋️ Model Training

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 10
- Validation Split: 10%
- Train-Test Split: 80-20

---

## ⚙️ How It Works

1. Images are loaded using OpenCV
2. All images are resized to (100x150)
3. Pixel values are normalized (0–1 scaling)
4. Data is split into training and testing sets
5. CNN model extracts features using convolution layers
6. Fully connected layers classify the output
7. Softmax layer gives probability for each class

---

## 📈 Results

- Training Accuracy: ~ (depends on dataset)
- Validation Accuracy: ~ (depends on dataset)

Accuracy graph is plotted using Matplotlib.

---

## 🧠 Why CNN?

CNNs are used because they:
- Automatically extract spatial features
- Work well with image data
- Reduce need for manual feature engineering
- Are highly effective for classification tasks

---

## ⚠️ Limitations

- Model performance depends heavily on dataset quality
- Can misclassify images with poor lighting or angles
- Limited to binary classification (Male/Female only)
- No real-time deployment yet

---

  ## 🚀 Future Improvements

- Integration of Transfer Learning (VGG16 / ResNet50)
- Real-time gender detection using webcam
- Deployment using Flask / FastAPI
- Mobile app integration
- Expand dataset for better generalization
- Add age detection along with gender classification

---

## 👨‍💻 Author

Full Stack Developer | AI & Machine Learning Enthusiast | Master's Aspirant in Information Technology

This project was built as part of my Deep Learning and Computer Vision learning journey to understand how CNNs improve upon traditional ANN architectures for image-based applications.
