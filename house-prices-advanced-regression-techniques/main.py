import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
categorical_columns = df.select_dtypes(include=['object']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Prepare the data
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the models
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

gb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
])

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the models
lr_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# Make predictions
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)
gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Evaluate the models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    print()

print("Training Set Results:")
evaluate_model(y_train, lr_train_pred, "Linear Regression")
evaluate_model(y_train, gb_train_pred, "Gradient Boosting")
evaluate_model(y_train, rf_train_pred, "Random Forest")


print("Test Set Results:")
evaluate_model(y_test, lr_test_pred, "Linear Regression")
evaluate_model(y_test, gb_test_pred, "Gradient Boosting")
evaluate_model(y_test, gb_test_pred, "Gradient Boosting")


# Cross-validation
def cv_rmse(model):
    cv_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-cv_scores)
    return rmse_scores

lr_cv_rmse = cv_rmse(lr_model)
gb_cv_rmse = cv_rmse(gb_model)
rf_cv_rmse = cv_rmse(rf_model)

print("Cross-validation RMSE:")
print(f"Linear Regression: {lr_cv_rmse.mean()} (+/- {lr_cv_rmse.std() * 2})")
print(f"Gradient Boosting: {gb_cv_rmse.mean()} (+/- {gb_cv_rmse.std() * 2})")
print(f"Random Forest: {rf_cv_rmse.mean()} (+/- {rf_cv_rmse.std() * 2})")


# Feature importance for Gradient Boosting
def plot_feature_importance(model, model_name):
    feature_importance = model.named_steps['regressor'].feature_importances_
    feature_names = (model.named_steps['preprocessor']
                     .named_transformers_['num'].get_feature_names_out(numerical_columns).tolist() +
                     model.named_steps['preprocessor']
                     .named_transformers_['cat'].get_feature_names_out(categorical_columns).tolist())
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Top 15 Feature Importance - {model_name}')
    plt.tight_layout()
    plt.show()

plot_feature_importance(gb_model, "Gradient Boosting")
plot_feature_importance(rf_model, "Random Forest")


# Prepare and make predictions on the test set
final_predictions_lr = lr_model.predict(test_df)
final_predictions_gb = gb_model.predict(test_df)
final_predictions_rf = rf_model.predict(test_df)

# Create submission files
submission_df_rf = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_predictions_rf})
submission_df_rf.to_csv('submission_rf.csv', index=False)
print("\nRandom Forest submission file created: submission_rf.csv")

submission_df_lr = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_predictions_lr})
submission_df_lr.to_csv('submission_lr.csv', index=False)
print("\nLinear Regression submission file created: submission_lr.csv")

submission_df_gb = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_predictions_gb})
submission_df_gb.to_csv('submission_gb.csv', index=False)
print("Gradient Boosting submission file created: submission_gb.csv")

# Scatter plot of predicted vs actual values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, rf_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest: Actual vs Predicted')

plt.subplot(1, 3, 2)
plt.scatter(y_test, lr_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')

plt.subplot(1, 3, 3)
plt.scatter(y_test, gb_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gradient Boosting: Actual vs Predicted')

plt.tight_layout()
plt.show()

# Residual plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
residuals_rf = y_test - rf_test_pred
plt.scatter(rf_test_pred, residuals_rf, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Random Forest: Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

plt.subplot(1, 3, 2)
residuals_lr = y_test - lr_test_pred
plt.scatter(lr_test_pred, residuals_lr, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Linear Regression: Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

plt.subplot(1, 3, 3)
residuals_gb = y_test - gb_test_pred
plt.scatter(gb_test_pred, residuals_gb, alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Gradient Boosting: Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()