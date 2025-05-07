
        # Model Evaluation Report: XGBoost

        **Generated on**: 2025-05-07 11:42:00

        ## Model Information
        - **Algorithm Type**: XGBClassifier
        - **Parameters**: 
        ```python
        {'objective': 'multi:softprob', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': None, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': None, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': None, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}
        ```

        ## Performance Metrics
        - **Accuracy**: 0.8442
        - **F1 Score (weighted)**: 0.8433
        - **Precision (weighted)**: 0.8445
        - **Recall (weighted)**: 0.8442

        ## Classification Report
        ```text
        0:
  precision: 0.9893
  recall: 0.9564
  f1-score: 0.9726
  support: 482.0000
1:
  precision: 0.9580
  recall: 0.9421
  f1-score: 0.9500
  support: 242.0000
2:
  precision: 0.9005
  recall: 0.9453
  f1-score: 0.9223
  support: 201.0000
3:
  precision: 0.9478
  recall: 0.9916
  f1-score: 0.9692
  support: 238.0000
4:
  precision: 0.5150
  recall: 0.4322
  f1-score: 0.4699
  support: 199.0000
5:
  precision: 0.5248
  recall: 0.6019
  f1-score: 0.5607
  support: 211.0000
accuracy: 0.8442
macro avg:
  precision: 0.8059
  recall: 0.8116
  f1-score: 0.8075
  support: 1573.0000
weighted avg:
  precision: 0.8445
  recall: 0.8442
  f1-score: 0.8433
  support: 1573.0000

        ```

        ## Visualizations

### Confusion Matrix
![Confusion Matrix](Machine Learning/Suprivised ML/FullRapport/visualizations/confusion_matrix_XGBoost.png)

### Feature Importance
![Feature Importance](Machine Learning/Suprivised ML/FullRapport/visualizations/feature_importance_XGBoost.png)

### Learning Curve
![Learning Curve](Machine Learning/Suprivised ML/FullRapport/visualizations/learning_curve_XGBoost.png)

## Recommendations
- This model shows good overall accuracy (>80%).
- Tree-based models like this typically handle complex decision boundaries well.
- They provide feature importance metrics that can help understand the problem.
- Class 4 (5.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
- Class 5 (6.0) has relatively poor performance (F1 < 0.7). Consider techniques to improve this specific class.
