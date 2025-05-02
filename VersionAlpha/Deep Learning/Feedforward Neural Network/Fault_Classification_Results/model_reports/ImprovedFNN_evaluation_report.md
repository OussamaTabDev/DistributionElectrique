# ImprovedFNN Model Evaluation Report

_Generated on 2025-05-02 11:25:10_

## Overall Performance

- **Accuracy**: 0.8093
- **Macro Average F1**: 0.7645
- **Weighted Average F1**: 0.7920

## Per-Class Performance

| Fault Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| No Fault | 0.9895 | 0.9895 | 0.9895 | 477 |
| LG | 0.9795 | 0.9317 | 0.9550 | 205 |
| LL BC | 0.9806 | 0.9619 | 0.9712 | 210 |
| LLG | 0.9885 | 0.7713 | 0.8665 | 223 |
| LLL | 0.4918 | 0.9459 | 0.6471 | 222 |
| LLLG | 0.2766 | 0.1102 | 0.1576 | 236 |
| **Macro Avg** | 0.7844 | 0.7851 | 0.7645 | 1573 |
| **Weighted Avg** | 0.8097 | 0.8093 | 0.7920 | 1573 |

## Performance Analysis

- Best performing class: **No Fault** with F1-score of 0.9895
- Worst performing class: **LLLG** with F1-score of 0.1576

- Most common confusion: 209 instances of **LLL** were misclassified as **LLG**

## Recommendations

- Consider model improvements to increase overall accuracy
- Focus on improving performance for LLLG fault type
- Consider collecting more training examples for LLLG
- Investigate why LLL is being confused with LLG
- Consider adding engineered features to better distinguish these classes
