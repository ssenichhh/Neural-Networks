# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates the development and evaluation of convolutional neural networks (CNNs) on the CIFAR-10 dataset. CIFAR-10 is a widely-used dataset for image classification tasks in computer vision and machine learning.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset includes everyday objects such as animals, vehicles, and other items.

## Dataset Characteristics

- **Number of classes:** 10
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Total images:** 60,000
- **Training images:** 50,000
- **Test images:** 10,000
- **Image size:** 32x32 pixels
- **Color channels:** 3 (RGB)

## Project Structure

1. **Develop a Convolutional Neural Network for CIFAR-10:**
    - Normalization of images
    - One-hot encoding of labels
    - CNN model with convolutional, pooling, flattening, dense, and dropout layers
    - Model compilation and training
    - Evaluation and visualization of results

2. **Experiment with Different Optimizers, Activation Functions, and Regularization Techniques:**
    - Use of different optimizers like SGD, RMSprop, Adam
    - Comparison of activation functions like ReLU, ELU
    - Implementation of regularization techniques like L2 regularization and Dropout

3. **Explore Different Network Architectures:**
    - Increase or decrease the number of layers
    - Adjust layer sizes

4. **Analysis and Visualization of Results:**
    - Plot training and validation loss and accuracy
    - Analyze performance and suggest improvements

## Usage

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

### Run the Project

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CIFAR-10-CNN.git
    ```

2. **Install the required packages:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```

3. **Execute the Jupyter Notebook:**
    - Open the notebook `CIFAR-10.ipynb` in Jupyter Notebook or Jupyter Lab.
    - Run the cells sequentially to train the model and evaluate its performance.

## Results

The project includes several CNN models trained with different configurations. Each model's performance is evaluated based on accuracy and loss on the test set. The results are visualized using Matplotlib.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research.
- The TensorFlow and Keras libraries are used for building and training the CNN models.
