# LSTM Model Training Report

_Generated on 2025-05-02 11:27:17_

## Model Description

Long Short-Term Memory recurrent network for sequence processing


                ## Long Short-Term Memory Network (LSTM)
                
                ### Architecture Overview
                A recurrent neural network architecture designed to process sequential data with the ability to remember important information over long periods.
                
                ### Key Features
                - **Structure**: LSTM layer (64 units) → Dropout (0.2) → Dense layer (32) → Output layer
                - **Memory Cells**: Contains specialized gates (input, forget, output) to control information flow
                - **Sequential Processing**: Processes input features as a sequence
                
                ### Strengths
                - Excellent at capturing temporal dependencies
                - Memory gates help with long-range dependencies
                - Prevents vanishing gradient problem better than standard RNNs
                
                ### Limitations
                - More computationally expensive than feedforward networks
                - More complex to tune and optimize
                - May be unnecessary if temporal relationships aren't important
            

## Model Architecture

```
Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 64)                  │          16,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_4 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_10 (Dense)                     │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_11 (Dense)                     │ (None, 7)                   │             231 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 57,623 (225.09 KB)
 Trainable params: 19,207 (75.03 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 38,416 (150.07 KB)

```

### Layer Distribution

- LSTM: 1
- Dropout: 1
- Dense: 2

## Model Complexity

- **Total Parameters**: 19,207
- **Trainable Parameters**: 19,207
- **Non-trainable Parameters**: 0
- **Memory Usage**: 75.03 KB

## Training Details

- **Training Time**: 45.00 seconds
- **Prediction Speed**: 4.4484 ms per sample
- **Epochs Trained**: 48

## Final Training Metrics

- **Training Accuracy**: 0.8456
- **Validation Accuracy**: 0.8308
- **Training Loss**: 0.2882
- **Validation Loss**: 0.3046

## Learning Curves

The model achieved its highest validation accuracy of 0.8372 at epoch 39. The model shows good fit, with high validation accuracy and minimal gap between training and validation performance.

## Architecture-Specific Notes

The LSTM network treats the input features as a sequence, potentially capturing temporal dependencies in the fault signals. This can be particularly useful if the order of features is meaningful.

## Recommendations

- The model is performing well; consider this architecture for deployment
- Fine-tune hyperparameters for potential incremental improvements
- Ensure model is robust to different fault conditions
