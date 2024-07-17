# SVHN Dataset Exploration and Modeling

This project demonstrates the process of loading, preprocessing, modeling, and evaluating the SVHN (Street View House Numbers) dataset using a pre-trained ResNet50 model.

## Overview

The SVHN dataset contains over 600,000 real-world images of house numbers captured from Google Street View. The dataset is commonly used for developing and evaluating machine learning models, particularly in the field of computer vision.

## Objectives

- Load and explore the SVHN dataset.
- Preprocess the images and labels.
- Build a transfer learning model using the ResNet50 architecture.
- Adapt the model to the SVHN dataset by modifying the top layers.
- Train the model using transfer learning techniques.
- Evaluate the model on the test dataset using various performance metrics (accuracy, precision, recall, F1-score).
- Visualize the training progress and results.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Neural-Networks/SVHN
    ```

2. **Navigate to the directory**:
    ```bash
    cd SVHN
    ```


## Code Structure

The project includes the following steps:

1. **Loading the SVHN Dataset**:
    - The dataset is loaded from `.mat` files.
    - Images and labels are extracted and reshaped appropriately.

2. **Preprocessing**:
    - Images are normalized to the range [0, 1].
    - Labels are converted to one-hot encoding.

3. **Model Building**:
    - A pre-trained ResNet50 model is loaded without the top layer.
    - Custom layers are added to adapt the model to the SVHN dataset.
    - The base model layers are frozen to retain pre-trained weights.

4. **Model Training**:
    - The model is compiled with the Adam optimizer and trained on the SVHN dataset.
    - Training and validation accuracy and loss are recorded.

5. **Evaluation**:
    - The trained model is evaluated on the test dataset.
    - Accuracy, precision, recall, and F1-score are calculated and printed.

6. **Visualization**:
    - Training and validation accuracy and loss curves are plotted.

## Results

The model's performance is evaluated using accuracy, precision, recall, and F1-score on the test dataset. Training and validation curves are plotted to visualize the model's progress.

## Conclusion

This project provides a comprehensive exploration of the SVHN dataset and demonstrates the effectiveness of transfer learning using ResNet50. Feel free to explore the code and experiment with different parameters to further improve the model's performance.

Happy coding!
