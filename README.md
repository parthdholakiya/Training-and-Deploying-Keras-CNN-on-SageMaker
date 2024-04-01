# Training and Deploying Keras CNN on SageMaker

## Overview

This repository contains code for training and deploying a Convolutional Neural Network (CNN) built using Keras on Amazon SageMaker. The CNN is trained on the MNIST dataset for handwritten digit recognition.

## Prerequisites

Before you begin, ensure you have the following prerequisites installed:

- Python 3.x
- AWS Account
- Boto3
- TensorFlow
- Keras
- SageMaker Python SDK

## Steps to Train and Deploy the Model

Follow these steps to train and deploy the Keras CNN model on SageMaker:

1. **Save the MNIST Dataset:** The MNIST dataset will be saved to disk in NPZ format.

2. **Upload MNIST Data to S3:** The saved dataset will be uploaded to an S3 bucket.

3. **Test the CNN Training Script Locally:** Test the training script locally on the notebook instance.

4. **Train the Model on SageMaker:** Train the CNN model on SageMaker using TensorFlow.

5. **Deploy the Model:** Deploy the trained model to a SageMaker endpoint.

6. **Make Predictions:** Use the deployed model to make predictions on new data.

7. **Cleanup:** Clean up the deployed endpoint to avoid unnecessary charges.

## Usage

Follow the instructions provided in the Jupyter Notebook or Python script to execute each step. Detailed explanations and comments are provided within the code for better understanding.

## Hyperparameter Tuning

To find the best hyperparameters for the model, you can use SageMaker's Automatic Model Tuning functionality. This will search through different combinations of hyperparameters to optimize the model's performance.

## Note

- Ensure that you have the necessary permissions to access SageMaker and S3 resources.
- Be mindful of costs associated with training and deploying models on SageMaker. Use appropriate instance types to avoid unnecessary charges.

## Author

This code was adapted from the following source: [Link to Source](https://aws.amazon.com/blogs/machine-learning/train-and-deploy-keras-models-with-tensorflow-and-apache-mxnet-on-amazon-sagemaker/)

## Acknowledgments

- Special thanks to the contributors of the original code and the AWS blog post mentioned above.
