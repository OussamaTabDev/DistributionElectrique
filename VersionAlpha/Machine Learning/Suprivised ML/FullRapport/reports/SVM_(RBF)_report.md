
        # Model Evaluation Report: SVM (RBF)

        **Generated on**: 2025-05-07 11:41:40

        ## Model Information
        - **Algorithm Type**: SVC
        - **Parameters**: 
        ```python
        {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8061
        - **F1 Score (weighted)**: 0.7903
        - **Precision (weighted)**: 0.8012
        - **Recall (weighted)**: 0.8061

        ## Classification Report
        ```text
        0:
  precision: 0.9146
  recall: 1.0000
  f1-score: 0.9554
  support: 482.0000
1:
  precision: 0.9188
  recall: 0.8884
  f1-score: 0.9034
  support: 242.0000
2:
  precision: 0.9938
  recall: 0.7960
  f1-score: 0.8840
  support: 201.0000
3:
  precision: 0.8920
  recall: 0.9370
  f1-score: 0.9139
  support: 238.0000
4:
  precision: 0.4810
  recall: 0.7638
  f1-score: 0.5903
  support: 199.0000
5:
  precision: 0.4235
  recall: 0.1706
  f1-score: 0.2432
  support: 211.0000
accuracy: 0.8061
macro avg:
  precision: 0.7706
  recall: 0.7593
  f1-score: 0.7484
  support: 1573.0000
weighted avg:
  precision: 0.8012
  recall: 0.8061
  f1-score: 0.7903
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_SVM_(RBF).png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_SVM_(RBF).png)

## Recommendations
- This model shows good overall accuracy (>80%).
- SVM models are effective in high-dimensional spaces.
- They work well with clear margin of separation in the data.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
