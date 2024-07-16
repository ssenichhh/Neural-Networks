# Gradient Descent with Digits Dataset

This project demonstrates how to implement a neural network from scratch and compare its performance with a pre-trained neural network model using the Digits dataset from scikit-learn. The project also explores different training parameters and analyzes their effects on the training process and model performance.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Neural Network Implementation](#neural-network-implementation)
- [Comparison with Scikit-learn MLPClassifier](#comparison-with-scikit-learn-mlpclassifier)
- [Optimization of Neural Network Architecture](#optimization-of-neural-network-architecture)
- [Experimenting with Training Parameters](#experimenting-with-training-parameters)
- [Conclusion](#conclusion)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ssenichhh/Neural-Networks.git
    cd Gradient Descent with Digits Dataset
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used in this project is the Digits dataset from scikit-learn, which contains images of hand-written digits. Each image is 8x8 pixels, and there are 10 classes representing digits 0 through 9.

## Neural Network Implementation

The project includes a custom implementation of a neural network using gradient descent. The neural network is trained to classify digits from the Digits dataset. Key functions include:

- `initialize_weights`: Initializes weights and biases.
- `sigmoid` and `sigmoid_derivative`: Activation function and its derivative.
- `forward_pass_multiclass`: Performs a forward pass through the network.
- `backward_pass_multiclass`: Performs a backward pass and updates weights.
- `compute_multiclass_loss`: Computes the loss.
- `train_model`: Trains the model using gradient descent.

## Comparison with Scikit-learn MLPClassifier

The performance of the custom neural network is compared with the `MLPClassifier` from scikit-learn. Key metrics for comparison include accuracy, precision, recall, and F1-score.

## Optimization of Neural Network Architecture

The architecture of the neural network is optimized by adding more hidden layers and changing activation functions. The project includes:

- `initialize_weights_modified`: Initializes weights for the modified architecture.
- `forward_pass_multiclass`: Modified forward pass.
- `backward_pass_multiclass`: Modified backward pass.
- `train_model_modified`: Trains the modified model.

## Experimenting with Training Parameters

The project explores the effects of different learning rates and numbers of epochs on the training process. The results are visualized using loss plots for different configurations.

## Conclusion

The project highlights the importance of choosing appropriate model architecture and training parameters for achieving high performance in machine learning tasks. The custom neural network and the pre-trained `MLPClassifier` from scikit-learn are compared, and various optimizations are applied to improve performance.

## Usage

1. Open the provided Jupyter notebook `Gradient_Descent_with_Digits_Dataset.ipynb`.
2. Run the notebook cells sequentially to execute the code.
3. Modify parameters such as learning rate and number of epochs in the notebook to experiment with different configurations.

## Acknowledgments

- The Digits dataset is provided by scikit-learn.
- The `MLPClassifier` is used from the scikit-learn library for comparison purposes.

