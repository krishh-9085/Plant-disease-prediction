# Plant Disease Recognition System 🌿

This repository contains a comprehensive solution for recognizing plant diseases using deep learning techniques. The project leverages TensorFlow for model training and prediction, along with Streamlit for an interactive web interface.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Web Application](#web-application)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Plant Disease Recognition System is designed to classify and identify diseases in plants based on images. The system incorporates the following components:

1. **Data Preprocessing:** The dataset is loaded and prepared for training and validation, ensuring images are resized and normalized appropriately.
2. **Model Building:** A Convolutional Neural Network (CNN) is constructed to learn the features of healthy and diseased plants from the images.
3. **Model Training and Evaluation:** The model is trained on the dataset and evaluated to measure its performance, including accuracy and loss metrics.
4. **Real-Time Predictions:** Users can upload images through a web application, where the model provides real-time predictions of plant diseases.

## Dataset

The model is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle. This dataset contains a diverse collection of images featuring both healthy and diseased plants, categorized into multiple classes, which enables effective training and validation of the model.

## Technologies Used

The project utilizes the following technologies:

- **Python**: The primary programming language for implementation.
- **TensorFlow**: For deep learning model development and training.
- **Keras**: A high-level neural networks API for building and training models.
- **Streamlit**: A framework for creating web applications for machine learning projects.
- **Matplotlib and Seaborn**: For visualizing training results and performance metrics.
- **OpenCV**: For image processing and manipulation.
- **NumPy**: For numerical operations and array manipulations.

## Getting Started

To get started with this project, ensure you have the necessary dependencies installed. This includes Python, TensorFlow, Keras, Streamlit, and other libraries used in the project. You can easily install the required packages using pip.

### Clone the Repository

Clone this repository to your local machine using Git. After cloning, you can navigate into the project directory to access the code and resources.

## Model Training and Evaluation

The model training process involves loading the dataset, splitting it into training and validation sets, and training the CNN to recognize various plant diseases. The model's performance is evaluated based on accuracy and loss metrics, which are recorded throughout the training process.

## Web Application

The web application provides an intuitive interface for users to upload images of plants. Once an image is uploaded, the trained model processes the image and predicts the disease, displaying the results on the screen. This enables quick and easy identification of plant diseases.

## Visualizations

The project includes visualizations to illustrate the model's training performance, such as accuracy and loss graphs over epochs. Additionally, confusion matrices are used to provide insights into model predictions and classification performance across different classes.

## Contributing

Contributions to the project are welcome! If you would like to contribute, feel free to fork the repository, make changes, and submit a pull request. Any feedback or suggestions for improvements are also appreciated.


