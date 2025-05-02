# LSTM Model Evaluation Report

_Generated on 2025-05-02 11:27:17_

## Overall Performance

- **Accuracy**: 0.8144
- **Macro Average F1**: 0.7715
- **Weighted Average F1**: 0.7983

## Per-Class Performance

| Fault Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| No Fault | 0.9655 | 0.9979 | 0.9814 | 477 |
| LG | 0.9217 | 0.9756 | 0.9479 | 205 |
| LL BC | 0.9596 | 0.9048 | 0.9314 | 210 |
| LLG | 0.8548 | 0.9507 | 0.9002 | 223 |
| LLL | 0.4841 | 0.6847 | 0.5672 | 222 |
| LLLG | 0.4951 | 0.2161 | 0.3009 | 236 |
| **Macro Avg** | 0.7801 | 0.7883 | 0.7715 | 1573 |
| **Weighted Avg** | 0.8048 | 0.8144 | 0.7983 | 1573 |

## Performance Analysis

- Best performing class: **No Fault** with F1-score of 0.9814
- Worst performing class: **LLLG** with F1-score of 0.3009

- Most common confusion: 153 instances of **LLL** were misclassified as **LLG**

## Recommendations

- Consider model improvements to increase overall accuracy
- Focus on improving performance for LLLG fault type
- Consider collecting more training examples for LLLG
- Investigate why LLL is being confused with LLG
- Consider adding engineered features to better distinguish these classes
