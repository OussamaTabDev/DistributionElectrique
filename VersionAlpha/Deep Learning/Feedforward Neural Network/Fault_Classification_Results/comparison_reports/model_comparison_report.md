# Model Comparison Report

_Generated on 2025-05-02 11:28:09_

## Model Rankings (by Weighted F1-Score)

| Rank | Model | Accuracy | Macro Avg F1 | Weighted Avg F1 | Training Time (s) | Prediction Speed (ms) | Parameters | Memory (KB) |
|------|-------|----------|--------------|-----------------|-------------------|----------------------|------------|------------|
| 1 | AdvancedCNN | 0.8163 | 0.7927 | 0.8158 | 38.01 | 2.9379 | 25,607 | 100.03 |
| 2 | GRU | 0.8125 | 0.7928 | 0.8144 | 46.25 | 5.5311 | 15,175 | 59.28 |
| 3 | FNN | 0.8436 | 0.7840 | 0.8078 | 23.26 | 2.4585 | 2,759 | 10.78 |
| 4 | LSTM | 0.8144 | 0.7715 | 0.7983 | 45.00 | 4.4484 | 19,207 | 75.03 |
| 5 | CNN | 0.8296 | 0.7723 | 0.7978 | 30.22 | 2.3577 | 4,615 | 18.03 |
| 6 | ImprovedFNN | 0.8093 | 0.7645 | 0.7920 | 28.77 | 2.1764 | 11,463 | 44.78 |

## Best Models by Category

- **highest accuracy score**: FNN (0.8436)
- **highest weighted F1-score**: AdvancedCNN (0.8158)
- **highest macro-average F1-score**: GRU (0.7928)
- **fastest training time**: FNN (23.26  (s))
- **fastest prediction speed**: ImprovedFNN (2.1764 ms per sample)
- **most efficient (fewest parameters)**: FNN (2,759 parameters)
- **smallest memory footprint**: FNN (10.78 (KB))

## Creative Model Analogies

- **FNN**: 🚗 Reliable Sedan - Gets you there efficiently but without flair
- **ImprovedFNN**: 🚙 SUV - More powerful version of the sedan
- **CNN**: 🛩️ Aircraft - Great for pattern recognition in spatial data
- **AdvancedCNN**: 🚀 Spaceship - Advanced version for complex patterns
- **LSTM**: 🕰️ Antique Clock - Excellent for temporal patterns but complex
- **GRU**: ⏱️ Modern Watch - Efficient time-aware processing

## Architecture Comparison

### FNN

Standard Feedforward Neural Network with two hidden layers

#### Layer Composition

- Dense: 3
- Dropout: 2

### ImprovedFNN

Enhanced Feedforward Neural Network with three hidden layers and increased complexity

#### Layer Composition

- Dense: 4
- Dropout: 2

### CNN

Convolutional Neural Network with single 1D convolution layer

#### Layer Composition

- Conv1D: 1
- MaxPooling1D: 1
- Flatten: 1
- Dense: 2

### AdvancedCNN

Complex CNN with multiple convolutional layers and global average pooling

#### Layer Composition

- Conv1D: 2
- MaxPooling1D: 1
- GlobalAveragePooling1D: 1
- Dense: 1

### LSTM

Long Short-Term Memory recurrent network for sequence processing

#### Layer Composition

- LSTM: 1
- Dropout: 1
- Dense: 2

### GRU

Gated Recurrent Unit network, a more efficient variant of LSTM

#### Layer Composition

- GRU: 1
- Dropout: 1
- Dense: 2

## Performance Trade-offs

### Accuracy vs. Speed

- The ImprovedFNN model is the fastest (prediction time: 2.1764 ms) but sacrifices 3.43% in accuracy compared to the most accurate model.
- The FNN model achieves the highest accuracy (0.8436) but is 1.1x slower than the fastest model.

### Complexity vs. Performance

- The AdvancedCNN model has 9.3x more parameters than the FNN model, resulting in a 0.99% improvement in F1-score.
- This suggests that the added complexity is 0.11% efficient in terms of F1-score improvement per parameter ratio increase.

## Overall Recommendations

### Best Overall Model: AdvancedCNN

The AdvancedCNN model provides the best balance of accuracy, F1-score, and efficiency for fault classification.

### Use Case Specific Recommendations

- **For resource-constrained environments**: FNN offers the smallest footprint.
- **For real-time applications**: ImprovedFNN provides the fastest inference time.
- **For maximum accuracy**: FNN achieves the highest classification accuracy.
- **For balanced performance**: AdvancedCNN offers the best weighted F1-score across all fault types.

