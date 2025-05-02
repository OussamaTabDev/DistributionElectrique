# GRU Model Evaluation Report

_Generated on 2025-05-02 11:28:08_

## Overall Performance

- **Accuracy**: 0.8125
- **Macro Average F1**: 0.7928
- **Weighted Average F1**: 0.8144

## Per-Class Performance

| Fault Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| No Fault | 0.9846 | 0.9371 | 0.9603 | 477 |
| LG | 0.9476 | 0.9707 | 0.9590 | 205 |
| LL BC | 0.8909 | 0.9333 | 0.9116 | 210 |
| LLG | 0.9817 | 0.9596 | 0.9705 | 223 |
| LLL | 0.4535 | 0.5270 | 0.4875 | 222 |
| LLLG | 0.4930 | 0.4449 | 0.4677 | 236 |
| **Macro Avg** | 0.7919 | 0.7955 | 0.7928 | 1573 |
| **Weighted Avg** | 0.8181 | 0.8125 | 0.8144 | 1573 |

## Performance Analysis

- Best performing class: **LLG** with F1-score of 0.9705
- Worst performing class: **LLLG** with F1-score of 0.4677

- Most common confusion: 130 instances of **LLL** were misclassified as **LLG**

## Recommendations

- Consider model improvements to increase overall accuracy
- Focus on improving performance for LLLG fault type
- Consider collecting more training examples for LLLG
- Investigate why LLL is being confused with LLG
- Consider adding engineered features to better distinguish these classes
