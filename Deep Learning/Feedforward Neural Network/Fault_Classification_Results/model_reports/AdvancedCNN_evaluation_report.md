# AdvancedCNN Model Evaluation Report

_Generated on 2025-05-02 11:26:28_

## Overall Performance

- **Accuracy**: 0.8163
- **Macro Average F1**: 0.7927
- **Weighted Average F1**: 0.8158

## Per-Class Performance

| Fault Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| No Fault | 0.9826 | 0.9497 | 0.9659 | 477 |
| LG | 0.8717 | 0.9610 | 0.9142 | 205 |
| LL BC | 0.9461 | 0.9190 | 0.9324 | 210 |
| LLG | 0.9345 | 0.9596 | 0.9469 | 223 |
| LLL | 0.5144 | 0.4820 | 0.4977 | 222 |
| LLLG | 0.4898 | 0.5085 | 0.4990 | 236 |
| **Macro Avg** | 0.7899 | 0.7966 | 0.7927 | 1573 |
| **Weighted Avg** | 0.8165 | 0.8163 | 0.8158 | 1573 |

## Performance Analysis

- Best performing class: **No Fault** with F1-score of 0.9659
- Worst performing class: **LLL** with F1-score of 0.4977

- Most common confusion: 111 instances of **LLG** were misclassified as **LLL**

## Recommendations

- Consider model improvements to increase overall accuracy
- Focus on improving performance for LLL fault type
- Consider collecting more training examples for LLL
- Investigate why LLG is being confused with LLL
- Consider adding engineered features to better distinguish these classes
