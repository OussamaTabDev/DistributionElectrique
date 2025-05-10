# AdvancedCNN Model Training Report

_Generated on 2025-05-02 11:26:28_

## Model Description

Complex CNN with multiple convolutional layers and global average pooling


                ## Advanced Convolutional Neural Network
                
                ### Architecture Overview
                A more sophisticated CNN with multiple convolutional layers and global pooling.
                
                ### Key Features
                - **Structure**: Conv1D layer (128 filters) → MaxPooling → Conv1D layer (64 filters) → GlobalAveragePooling → Output layer
                - **Filter Sizes**: 3 units with padding='same' for feature preservation
                - **Pooling**: Global average pooling reduces parameters while maintaining feature importance
                
                ### Strengths
                - Deeper architecture can capture more complex patterns
                - Global average pooling reduces overfitting
                - Padding preserves spatial dimensions better
                
                ### Limitations
                - More computationally intensive
                - May be more than needed for simpler classification tasks
                - Requires more data to generalize effectively
            

## Model Architecture

```
Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv1d_1 (Conv1D)                    │ (None, 6, 128)              │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling1d_1 (MaxPooling1D)       │ (None, 3, 128)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1d_2 (Conv1D)                    │ (None, 3, 64)               │          24,640 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling1d             │ (None, 64)                  │               0 │
│ (GlobalAveragePooling1D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_9 (Dense)                      │ (None, 7)                   │             455 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 76,823 (300.09 KB)
 Trainable params: 25,607 (100.03 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 51,216 (200.07 KB)

```

### Layer Distribution

- Conv1D: 2
- MaxPooling1D: 1
- GlobalAveragePooling1D: 1
- Dense: 1

## Model Complexity

- **Total Parameters**: 25,607
- **Trainable Parameters**: 25,607
- **Non-trainable Parameters**: 0
- **Memory Usage**: 100.03 KB

## Training Details

- **Training Time**: 38.01 seconds
- **Prediction Speed**: 2.9379 ms per sample
- **Epochs Trained**: 50

## Final Training Metrics

- **Training Accuracy**: 0.8327
- **Validation Accuracy**: 0.8206
- **Training Loss**: 0.3062
- **Validation Loss**: 0.2940

## Learning Curves

The model achieved its highest validation accuracy of 0.8206 at epoch 49. The model shows good fit, with high validation accuracy and minimal gap between training and validation performance.

## Architecture-Specific Notes

The advanced CNN with multiple convolutional layers can detect hierarchical features in fault signals. The global average pooling helps make the model more robust to variations in the fault patterns.

## Recommendations

- The model is performing well; consider this architecture for deployment
- Fine-tune hyperparameters for potential incremental improvements
- Ensure model is robust to different fault conditions
