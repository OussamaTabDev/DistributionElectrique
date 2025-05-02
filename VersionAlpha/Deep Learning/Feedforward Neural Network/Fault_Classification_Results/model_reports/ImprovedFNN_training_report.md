# ImprovedFNN Model Training Report

_Generated on 2025-05-02 11:25:09_

## Model Description

Enhanced Feedforward Neural Network with three hidden layers and increased complexity


                ## Improved Feedforward Neural Network
                
                ### Architecture Overview
                An enhanced version of the standard FNN with deeper architecture and improved regularization.
                
                ### Key Features
                - **Structure**: Input layer → Hidden layer 1 (128 neurons) → Hidden layer 2 (64 neurons) → Hidden layer 3 (32 neurons) → Output layer
                - **Activation**: ReLU in hidden layers, Softmax in output layer
                - **Regularization**: Stronger dropout layers (0.3) to prevent overfitting
                
                ### Strengths
                - Greater capacity to learn complex patterns
                - Better regularization prevents overfitting on smaller datasets
                - Additional layer allows hierarchical feature extraction
                
                ### Limitations
                - More parameters to train compared to standard FNN
                - Longer training time
                - May be prone to vanishing gradients in deeper configurations
            

## Model Architecture

```
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                      │ (None, 128)                 │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_5 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_6 (Dense)                      │ (None, 7)                   │             231 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 34,391 (134.34 KB)
 Trainable params: 11,463 (44.78 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 22,928 (89.57 KB)

```

### Layer Distribution

- Dense: 4
- Dropout: 2

## Model Complexity

- **Total Parameters**: 11,463
- **Trainable Parameters**: 11,463
- **Non-trainable Parameters**: 0
- **Memory Usage**: 44.78 KB

## Training Details

- **Training Time**: 28.77 seconds
- **Prediction Speed**: 2.1764 ms per sample
- **Epochs Trained**: 50

## Final Training Metrics

- **Training Accuracy**: 0.8422
- **Validation Accuracy**: 0.8639
- **Training Loss**: 0.2713
- **Validation Loss**: 0.2261

## Learning Curves

The model achieved its highest validation accuracy of 0.8690 at epoch 47. The model shows good fit, with high validation accuracy and minimal gap between training and validation performance.

## Architecture-Specific Notes

The improved FNN with additional layers and neurons can capture more complex patterns than the standard FNN. This architecture represents a good balance between model complexity and performance.

## Recommendations

- The model is performing well; consider this architecture for deployment
- Fine-tune hyperparameters for potential incremental improvements
- Ensure model is robust to different fault conditions
