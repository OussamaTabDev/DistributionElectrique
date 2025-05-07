
        # Model Evaluation Report: SVM (Linear)

        **Generated on**: 2025-05-07 11:41:21

        ## Model Information
        - **Algorithm Type**: SVC
        - **Parameters**: 
        ```python
        {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.6128
        - **F1 Score (weighted)**: 0.5626
        - **Precision (weighted)**: 0.6378
        - **Recall (weighted)**: 0.6128

        ## Classification Report
        ```text
        0:
  precision: 0.6603
  recall: 1.0000
  f1-score: 0.7954
  support: 482.0000
1:
  precision: 0.7515
  recall: 0.5248
  f1-score: 0.6180
  support: 242.0000
2:
  precision: 0.8621
  recall: 0.3731
  f1-score: 0.5208
  support: 201.0000
3:
  precision: 0.4650
  recall: 0.8361
  f1-score: 0.5976
  support: 238.0000
4:
  precision: 0.5714
  recall: 0.0603
  f1-score: 0.1091
  support: 199.0000
5:
  precision: 0.5000
  recall: 0.3270
  f1-score: 0.3954
  support: 211.0000
accuracy: 0.6128
macro avg:
  precision: 0.6350
  recall: 0.5202
  f1-score: 0.5061
  support: 1573.0000
weighted avg:
  precision: 0.6378
  recall: 0.6128
  f1-score: 0.5626
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_SVM_(Linear).png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_SVM_(Linear).png)

## Recommendations
- This model's accuracy could be improved.
- SVM models are effective in high-dimensional spaces.
- They work well with clear margin of separation in the data.
- Class 1 (1.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 2 (3.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 3 (4.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
