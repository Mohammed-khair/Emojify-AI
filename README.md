# Emojify-AI Notebook Readme

This notebook demonstrates the process of building an Emojify model using word embeddings and LSTM layers to predict appropriate emojis for input sentences. The model utilizes pre-trained GloVe word vectors for word representations and employs Keras for implementation.

## Acknowledgements

In this notebook, we'll be using GloVe vectors for our word embeddings. The GloVe vectors were introduced in the following paper:

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).

Parts of this notebook was done as an exercise in the Coursera course: [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models).

## Contents

1. Introduction
2. Loading Word Vectors
3. Loading Training and Testing Sets
4. Maximum Sequence Length
5. Embedding Layer
6. Building the Emojify Model
7. Model Compilation and Training
8. Model Evaluation on Testing Set
9. Trying the Model with Custom Inputs

## 1. Introduction

This notebook showcases the creation of an Emojify model that predicts emojis for input sentences. It utilizes LSTM layers and GloVe word vectors for effective text analysis and understanding.

## 2. Loading Word Vectors

The GloVe vectors are loaded into the model to represent words as vectors. These vectors are crucial for the model's understanding of the semantic meaning of words.

## 3. Loading Training and Testing Sets

The training and testing data are loaded from CSV files (`train_emoji.csv` and `test_emoji.csv`). These datasets contain sentences and their corresponding labels (emojis).

## 4. Maximum Sequence Length

The maximum sequence length in the training and test sets is determined. This length is used for padding or truncating sentences during preprocessing.

## 5. Embedding Layer

An Embedding layer is created to convert word indices to embedding vectors. The layer's weights are initialized using pre-trained GloVe word vectors.

## 6. Building the Emojify Model

The Emojify model is built using Keras. It consists of an embedding layer, two LSTM layers, dropout layers for regularization, and a dense layer with softmax activation for multi-class classification.

## 7. Model Compilation and Training

The model is compiled with categorical cross-entropy loss and the Adam optimizer. It is then trained on the training data for a specified number of epochs and batch size.

## 8. Model Evaluation on Testing Set

The model's performance is evaluated on the testing set using accuracy as the metric. The test accuracy is printed as the output.

## 9. Trying the Model with Custom Inputs

The model is tested with custom input sentences to demonstrate its ability to predict emojis for user-generated text.

Feel free to experiment with the notebook to explore and understand the process of building an Emojify model using word embeddings and LSTM layers.
