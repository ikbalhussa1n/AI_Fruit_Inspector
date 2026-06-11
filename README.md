# 🍎 Fruit Freshness Classifier

A deep learning-based image classification system that detects whether a fruit is **fresh or rotten** using a Convolutional Neural Network (CNN).  
The model is deployed using **Streamlit**, allowing users to upload an image and get real-time predictions.

---

<img width="1145" height="1189" alt="image" src="https://github.com/user-attachments/assets/4627d994-5209-4903-ba53-d66486ca0e73" />


## 📌 Table of Contents
- Project Overview
- Dataset
- Model Architecture
- Setup and Installation
- Running the App Locally
- Deployment
- Usage

---

## 📖 Project Overview

This project builds an AI-powered system to classify fruit freshness from images.  
It can be applied in:

- Supermarket quality control 🛒  
- Automated sorting systems ⚙️  
- Personal food inspection 🍏  

The system uses a trained CNN model to analyze fruit images and classify them as **Fresh** or **Rotten**.

---

## 📊 Dataset

The model is trained on:

👉 `sriramr/fruits-fresh-and-rotten-for-classification` (KaggleHub dataset)

### Classes:
- Fresh Apples 🍏
- Fresh Bananas 🍌
- Fresh Oranges 🍊
- Rotten Apples 🍎
- Rotten Bananas 🍌
- Rotten Oranges 🍊

Images are preprocessed and augmented before training to improve generalization.

---

## 🧠 Model Architecture

The model is built using **TensorFlow / Keras** and includes:

- Conv2D layers (feature extraction)
- MaxPooling2D layers (downsampling)
- Flatten layer (convert features to vector)
- Dense layers (classification)
- Dropout layer (regularization)

### 🔄 Data Augmentation:
- Random flip
- Random rotation
- Random zoom
- Random contrast

---

## ⚙️ Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI_Fruit_Inspector.git
cd AI_Fruit_Inspector

###  2. Install dependencies
```bash
pip install -r requirements.txt
    ```

The `requirements.txt` file typically contains:
    ```
    streamlit
    tensorflow
    numpy
    pillow
    gdown
    ```

### How to Run the Streamlit App

Once the dependencies are installed and you have the `streamlit_app.py` file (generated in the Colab notebook):

1.  **Ensure `main.py` is in your working directory.**
2.  **Run the Streamlit application** using the following command in your terminal:

    ```bash
    streamlit run main.py
    ```

3.  This will open the application in your default web browser. If it doesn't open automatically, navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

## Deployment

The Streamlit application is also deployed and accessible online:

**Deployment Link:** [https://aifruitinspector-zvarujnmxczfmy95dhoffs.streamlit.app/](https://aifruitinspector-zvarujnmxczfmy95dhoffs.streamlit.app/)

## Usage

1.  Open the Streamlit application (either locally or via the deployment link).
2.  You will see an option to upload an image.
3.  Upload an image of a fruit (e.g., apple, banana, orange).
4.  The application will display the uploaded image and then predict whether the fruit is "Fresh" or "Rotten" along with a confidence score.
```
