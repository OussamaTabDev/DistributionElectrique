
        # Model Evaluation Report: Decision Tree

        **Generated on**: 2025-05-07 11:40:08

        ## Model Information
        - **Algorithm Type**: DecisionTreeClassifier
        - **Parameters**: 
        ```python
        {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'random_state': None, 'splitter': 'best'}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8131
        - **F1 Score (weighted)**: 0.8160
        - **Precision (weighted)**: 0.8228
        - **Recall (weighted)**: 0.8131

        ## Classification Report
        ```text
        0:
  precision: 0.9837
  recall: 0.8776
  f1-score: 0.9276
  support: 482.0000
1:
  precision: 0.9871
  recall: 0.9504
  f1-score: 0.9684
  support: 242.0000
2:
  precision: 0.7839
  recall: 0.9204
  f1-score: 0.8467
  support: 201.0000
3:
  precision: 0.9634
  recall: 0.9958
  f1-score: 0.9793
  support: 238.0000
4:
  precision: 0.4398
  recall: 0.4221
  f1-score: 0.4308
  support: 199.0000
5:
  precision: 0.5063
  recall: 0.5687
  f1-score: 0.5357
  support: 211.0000
accuracy: 0.8131
macro avg:
  precision: 0.7774
  recall: 0.7892
  f1-score: 0.7814
  support: 1573.0000
weighted avg:
  precision: 0.8228
  recall: 0.8131
  f1-score: 0.8160
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_Decision_Tree.png)

### Feature Importance
![Feature Importance](Machine Learning/Suprivised ML/FullRapport/visualizations/feature_importance_Decision_Tree.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_Decision_Tree.png)

### Decision Tree
![Decision Tree](Machine Learning/Suprivised ML/FullRapport/visualizations/decision_tree_Decision_Tree.png)

## Recommendations
- This model shows good overall accuracy (>80%).
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
