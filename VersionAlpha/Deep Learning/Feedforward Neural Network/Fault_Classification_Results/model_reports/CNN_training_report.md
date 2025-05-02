# CNN Model Training Report

_Generated on 2025-05-02 11:25:44_

## Model Description

Convolutional Neural Network with single 1D convolution layer


                ## Convolutional Neural Network (CNN)
                
                ### Architecture Overview
                A neural network using convolutional filters to extract spatial features from input data.
                
                ### Key Features
                - **Structure**: Conv1D layer (64 filters) → MaxPooling → Flatten → Dense layer (32) → Output layer
                - **Filter Size**: 3 units (kernel_size=3)
                - **Pooling**: Max pooling with stride 2 to reduce dimensionality
                
                ### Strengths
                - Effective at capturing local patterns and spatial relationships
                - Parameter sharing reduces model size
                - Translation invariant (can detect patterns regardless of position)
                
                ### Limitations
                - Less effective for purely numerical data without spatial relationships
                - Simpler architecture may miss complex temporal dynamics
            

## Model Architecture

```
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv1d (Conv1D)                      │ (None, 4, 64)               │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d (MaxPooling1D)         │ (None, 2, 64)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_7 (Dense)                      │ (None, 32)                  │           4,128 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_8 (Dense)                      │ (None, 7)                   │             231 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 13,847 (54.09 KB)
 Trainable params: 4,615 (18.03 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 9,232 (36.07 KB)

```

### Layer Distribution

- Conv1D: 1
- MaxPooling1D: 1
- Flatten: 1
- Dense: 2

## Model Complexity

- **Total Parameters**: 4,615
- **Trainable Parameters**: 4,615
- **Non-trainable Parameters**: 0
- **Memory Usage**: 18.03 KB

## Training Details

- **Training Time**: 30.22 seconds
- **Prediction Speed**: 2.3577 ms per sample
- **Epochs Trained**: 50

## Final Training Metrics

- **Training Accuracy**: 0.8503
- **Validation Accuracy**: 0.8518
- **Training Loss**: 0.2860
- **Validation Loss**: 0.2800

## Learning Curves

The model achieved its highest validation accuracy of 0.8518 at epoch 50. The model shows good fit, with high validation accuracy and minimal gap between training and validation performance.

## Architecture-Specific Notes

The convolutional architecture extracts spatial features from the input signals, which can be beneficial for detecting fault patterns that manifest as specific signal shapes or transitions.

## Recommendations

- The model is performing well; consider this architecture for deployment
- Fine-tune hyperparameters for potential incremental improvements
- Ensure model is robust to different fault conditions
