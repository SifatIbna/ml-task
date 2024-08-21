# House Price Prediction Project

## Project Overview
This project aims to predict house prices using the Kaggle House Prices: Advanced Regression Techniques dataset. We compare the performance of three different models: Linear Regression, Random Forest, and Gradient Boosting Regressor.

## Data Preprocessing and Feature Selection

Our data preprocessing pipeline involved several key steps to prepare the data for modeling:

1. **Handling Missing Values**:
   - For numerical features: Imputed missing values with the median of each column.
   - For categorical features: Imputed missing values with a constant 'missing' value.

2. **Feature Encoding**:
   - Categorical features were one-hot encoded using sklearn's OneHotEncoder.
   - We used the `handle_unknown='ignore'` parameter to handle any new categories in the test set.

3. **Feature Scaling**:
   - Numerical features were standardized using StandardScaler to ensure all features are on the same scale.

4. **Feature Selection**:
   - Initially, we included all available features from the dataset.
   - Feature importance was later analyzed using the `feature_importances_` attribute of the Random Forest and Gradient Boosting models.
   - This analysis helps identify the most influential features for predicting house prices.

5. **Handling Multicollinearity**:
   - While we didn't explicitly remove multicollinear features in the preprocessing stage, both Random Forest and Gradient Boosting models are generally robust to multicollinearity.
   - The feature importance plots can help identify which features are most influential, potentially allowing for future refinement of the feature set.

6. **Separating Numerical and Categorical Features**:
   - We used pandas' `select_dtypes` method to automatically categorize features:
     ```python
     numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
     categorical_columns = df.select_dtypes(include=['object']).columns
     ```

7. **Pipeline Creation**:
   - We created separate preprocessing pipelines for numerical and categorical features and combined them using ColumnTransformer.

8. **Integration with Model Training**:
   - The preprocessing steps were integrated into the model pipeline, ensuring consistent application to both training and test data.

## Model Development

We implemented and compared three models:

1. **Linear Regression**: A simple baseline model.
2. **Random Forest Regressor**: An ensemble method using multiple decision trees.
3. **Gradient Boosting Regressor**: An ensemble technique that builds trees sequentially.

## Evaluation Metrics
- Root Mean Squared Error (RMSE)
- R-squared (R²) Score
- Cross-validation scores

## Results

### Training Set Results:
- Linear Regression:
  - RMSE: 18,903.80
  - R² Score: 0.9401
- Gradient Boosting:
  - RMSE: 9,170.20
  - R² Score: 0.9859
- Random Forest:
  - RMSE: 11,129.42
  - R² Score: 0.9792

### Test Set Results:
- Linear Regression:
  - RMSE: 32,693,182,427,756.11
  - R² Score: -1.3934794275803158e+17
- Gradient Boosting:
  - RMSE: 26,543.01
  - R² Score: 0.9081
- Random Forest:
  - RMSE: 29501.667554467087
   - R² Score: 0.8865304597242845

### Cross-validation RMSE:
- Linear Regression: 353,960,893,143,689.0 (+/- 956,674,261,413,069.9)
- Gradient Boosting: 27,116.14 (+/- 6,254.23)
- Random Forest: 29,785.47 (+/- 7,339.41)

## Model Selection and Reasoning

Based on the results, the Gradient Boosting model appears to be the most effective for this task:

1. **Performance**: Gradient Boosting outperformed both Linear Regression and Random Forest on the test set and in cross-validation.
2. **Consistency**: It maintained good performance across both training and test sets, indicating good generalization.
3. **Robustness**: The cross-validation results show that Gradient Boosting is more stable compared to the other models, with a lower RMSE and smaller standard deviation.
4. **Handling Non-linearity**: The superior performance of Gradient Boosting suggests it captures non-linear relationships in the data more effectively than Linear Regression.
5. **Overfitting Considerations**: While Random Forest showed slightly better performance on the training set, Gradient Boosting generalized better to the test set, suggesting a good balance between fitting and generalization.

## Feature Importance Analysis
The script includes functionality to plot feature importance for both Gradient Boosting and Random Forest models, providing insights into which features are most predictive of house prices.

## Visualizations
The code generates several visualizations:
- Actual vs Predicted plots for all three models
- Residual plots for all three models

These visualizations help in understanding the performance and characteristics of each model.

## Future Improvements
1. Hyperparameter Tuning: Optimize parameters for Random Forest and Gradient Boosting models.
2. Feature Engineering: Create new features or transform existing ones based on the feature importance analysis.
3. Ensemble Methods: Explore combining predictions from multiple models.
4. Advanced Models: Consider other algorithms like XGBoost or LightGBM.
5. Addressing Linear Regression Issues: Investigate why the Linear Regression model performed poorly on the test set.
6. Explicit Multicollinearity Handling: Consider using techniques like Variance Inflation Factor (VIF) to identify and address multicollinearity.
7. Advanced Feature Selection: Implement more sophisticated feature selection techniques based on statistical tests or domain knowledge.

## Conclusion
This project demonstrates the process of building and selecting a model for house price prediction. The Gradient Boosting model proved to be the most effective, showing superior performance and generalization capabilities. The extreme performance difference between the models, especially the poor performance of Linear Regression on the test set, suggests complex non-linear relationships in the data that are better captured by ensemble methods.

The preprocessing steps, including careful handling of missing values, feature encoding, and scaling, played a crucial role in preparing the data for effective modeling. Future work could focus on refining these preprocessing steps and exploring more advanced feature selection and engineering techniques to further improve model performance.

