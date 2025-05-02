# CNN Model Evaluation Report

_Generated on 2025-05-02 11:25:45_

## Overall Performance

- **Accuracy**: 0.8296
- **Macro Average F1**: 0.7723
- **Weighted Average F1**: 0.7978

## Per-Class Performance

| Fault Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| No Fault | 0.9793 | 0.9895 | 0.9844 | 477 |
| LG | 0.9387 | 0.9707 | 0.9544 | 205 |
| LL BC | 0.9800 | 0.9333 | 0.9561 | 210 |
| LLG | 0.9900 | 0.8924 | 0.9387 | 223 |
| LLL | 0.4922 | 0.9910 | 0.6577 | 222 |
| LLLG | 0.6129 | 0.0805 | 0.1423 | 236 |
| **Macro Avg** | 0.8322 | 0.8096 | 0.7723 | 1573 |
| **Weighted Avg** | 0.8519 | 0.8296 | 0.7978 | 1573 |

## Performance Analysis

- Best performing class: **No Fault** with F1-score of 0.9844
- Worst performing class: **LLLG** with F1-score of 0.1423

- Most common confusion: 216 instances of **LLL** were misclassified as **LLG**

## Recommendations

- Consider model improvements to increase overall accuracy
- Focus on improving performance for LLLG fault type
- Consider collecting more training examples for LLLG
- Investigate why LLL is being confused with LLG
- Consider adding engineered features to better distinguish these classes
