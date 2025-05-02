# Stanford Cars Image Classification Project

# [📦 Download Stanford Cars Dataset & model.keras (Google Drive)](https://drive.google.com/drive/folders/1GW2gUQw7waEc0WgCe9ERmOMABtDIYQFU?usp=sharing)

## Overview

This project focuses on building a Convolutional Neural Network (CNN) to classify car images using the Stanford Cars dataset. The dataset includes 196 classes of cars with over 8,000 training images. Initially, the data was unorganized, so I used annotation files (`cars_train_annos.mat` and `cars_meta.mat`) to reorganize the training images into folders by class. The data was then preprocessed using TensorFlow’s image loading utilities and normalized to improve training performance. The model was trained over 10 epochs using image sizes of 180x180 and a batch size of 32.

## Modeling Approach

A CNN was selected for this task because convolutional layers are highly effective at identifying visual patterns in images. The model consists of three convolutional blocks followed by dense layers, using ReLU activation functions and max-pooling for feature extraction. The final layer uses a softmax output over 196 classes. I compiled the model using Adam optimizer and Sparse Categorical Crossentropy loss. After training, the model achieved an accuracy of approximately **91%** and a loss of **0.34** on the training dataset. The trained model was saved as `car_model.keras` for future predictions.

