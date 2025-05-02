# GRU Model Training Report

_Generated on 2025-05-02 11:28:08_

## Model Description

Gated Recurrent Unit network, a more efficient variant of LSTM


                ## Gated Recurrent Unit Network (GRU)
                
                ### Architecture Overview
                A streamlined version of LSTM with fewer parameters but similar effectiveness for many tasks.
                
                ### Key Features
                - **Structure**: GRU layer (64 units) → Dropout (0.2) → Dense layer (32) → Output layer
                - **Gating Mechanism**: Uses update and reset gates (simpler than LSTM)
                - **Sequential Processing**: Processes input features as a sequence
                
                ### Strengths
                - Fewer parameters than LSTM (more efficient)
                - Still effective at capturing sequential patterns
                - Often trains faster than LSTM
                
                ### Limitations
                - May be less effective than LSTM for very long sequences
                - Still more computationally expensive than non-recurrent models
                - Benefits may be minimal if data has no sequential nature
            

## Model Architecture

```
Model: "sequential_5"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ gru (GRU)                            │ (None, 64)                  │          12,864 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_5 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_12 (Dense)                     │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_13 (Dense)                     │ (None, 7)                   │             231 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 45,527 (177.84 KB)
 Trainable params: 15,175 (59.28 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 30,352 (118.57 KB)

```

### Layer Distribution

- GRU: 1
- Dropout: 1
- Dense: 2

## Model Complexity

- **Total Parameters**: 15,175
- **Trainable Parameters**: 15,175
- **Non-trainable Parameters**: 0
- **Memory Usage**: 59.28 KB

## Training Details

- **Training Time**: 46.25 seconds
- **Prediction Speed**: 5.5311 ms per sample
- **Epochs Trained**: 50

## Final Training Metrics

- **Training Accuracy**: 0.8675
- **Validation Accuracy**: 0.8372
- **Training Loss**: 0.2328
- **Validation Loss**: 0.2463

## Learning Curves

The model achieved its highest validation accuracy of 0.8454 at epoch 17. The model shows good fit, with high validation accuracy and minimal gap between training and validation performance.

## Architecture-Specific Notes

The GRU network provides similar sequential processing capabilities to LSTM but with fewer parameters, potentially offering faster training and inference times while maintaining good performance.

## Recommendations

- The model is performing well; consider this architecture for deployment
- Fine-tune hyperparameters for potential incremental improvements
- Ensure model is robust to different fault conditions
