# FNN Model Training Report

_Generated on 2025-05-02 11:24:35_

## Model Description

Standard Feedforward Neural Network with two hidden layers


                ## Feedforward Neural Network (FNN)
                
                ### Architecture Overview
                The FNN is the simplest deep learning architecture, consisting of fully connected layers where each neuron connects to every neuron in the previous and next layer.
                
                ### Key Features
                - **Structure**: Input layer → Hidden layer 1 (64 neurons) → Hidden layer 2 (32 neurons) → Output layer
                - **Activation**: ReLU in hidden layers, Softmax in output layer
                - **Regularization**: Dropout layers (0.2) to prevent overfitting
                
                ### Strengths
                - Simple and computationally efficient
                - Works well for tabular data with clear feature relationships
                - Quick to train and deploy
                
                ### Limitations
                - Limited ability to capture complex spatial or temporal patterns
                - May require more feature engineering
                - Often requires more parameters for the same performance as specialized architectures
            

## Model Architecture

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 64)                  │             448 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 7)                   │             231 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 8,279 (32.34 KB)
 Trainable params: 2,759 (10.78 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 5,520 (21.57 KB)

```

### Layer Distribution

- Dense: 3
- Dropout: 2

## Model Complexity

- **Total Parameters**: 2,759
- **Trainable Parameters**: 2,759
- **Non-trainable Parameters**: 0
- **Memory Usage**: 10.78 KB

## Training Details

- **Training Time**: 23.26 seconds
- **Prediction Speed**: 2.4585 ms per sample
- **Epochs Trained**: 50

## Final Training Metrics

- **Training Accuracy**: 0.8399
- **Validation Accuracy**: 0.8601
- **Training Loss**: 0.2922
- **Validation Loss**: 0.2402

## Learning Curves

The model achieved its highest validation accuracy of 0.8601 at epoch 50. The model shows good fit, with high validation accuracy and minimal gap between training and validation performance.

## Architecture-Specific Notes

This standard feedforward neural network provides a good baseline for fault classification. Consider this architecture when computational resources are limited or when a simple, interpretable model is desired.

## Recommendations

- The model is performing well; consider this architecture for deployment
- Fine-tune hyperparameters for potential incremental improvements
- Ensure model is robust to different fault conditions
