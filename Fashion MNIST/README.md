# Autoencoder Experiments with Fashion MNIST

This project demonstrates the implementation and comparison of autoencoders using different optimizers and activation functions on the Fashion MNIST dataset.

## Overview

Fashion MNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## Objectives

- Understand the concept and architecture of autoencoders.
- Develop and train autoencoders on the Fashion MNIST dataset.
- Compare the performance of autoencoders using different optimizers (adam, sgd, rmsprop) and activation functions (relu, elu).
- Visualize the reconstructed images and calculate the Mean Squared Error (MSE).

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repository/autoencoder-fashion-mnist.git
    ```

2. **Navigate to the directory**:
    ```bash
    cd autoencoder-fashion-mnist
    ```

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook Autoencoder_Fashion_MNIST.ipynb
    ```

## Results

Each autoencoder model is trained for 10 epochs, and the results are visualized by plotting the original and reconstructed images, as well as the training and validation loss curves. The Mean Squared Error (MSE) is also calculated and displayed.

## Conclusion

The project provides a comprehensive comparison of different autoencoder models on the Fashion MNIST dataset, helping to understand the impact of various optimizers and activation functions on the model's performance.

Feel free to explore the code and modify the parameters to further improve the autoencoder models.

Happy coding!
