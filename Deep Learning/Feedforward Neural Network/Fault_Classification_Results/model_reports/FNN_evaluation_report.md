# FNN Model Evaluation Report

_Generated on 2025-05-02 11:24:35_

## Overall Performance

- **Accuracy**: 0.8436
- **Macro Average F1**: 0.7840
- **Weighted Average F1**: 0.8078

## Per-Class Performance

| Fault Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| No Fault | 0.9854 | 0.9874 | 0.9864 | 477 |
| LG | 0.9755 | 0.9707 | 0.9731 | 205 |
| LL BC | 0.9663 | 0.9571 | 0.9617 | 210 |
| LLG | 0.9776 | 0.9776 | 0.9776 | 223 |
| LLL | 0.5000 | 0.9910 | 0.6647 | 222 |
| LLLG | 0.9000 | 0.0763 | 0.1406 | 236 |
| **Macro Avg** | 0.8841 | 0.8267 | 0.7840 | 1573 |
| **Weighted Avg** | 0.8991 | 0.8436 | 0.8078 | 1573 |

## Performance Analysis

- Best performing class: **No Fault** with F1-score of 0.9864
- Worst performing class: **LLLG** with F1-score of 0.1406

- Most common confusion: 214 instances of **LLL** were misclassified as **LLG**

## Recommendations

- Consider model improvements to increase overall accuracy
- Focus on improving performance for LLLG fault type
- Consider collecting more training examples for LLLG
- Investigate why LLL is being confused with LLG
- Consider adding engineered features to better distinguish these classes
