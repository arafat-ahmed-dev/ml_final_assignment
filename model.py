# Mobile Price Classification

# importing libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# ===========================================
# 1. Data Loading
# ===========================================
print("=" * 50)
print("1. DATA LOADING")
print("=" * 50)

# loading the data
train_data = pd.read_csv('mobile_price_classsification_train.csv')
test_data = pd.read_csv('mobile_price_classsification_test1.csv')

# checking the shape
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# first 5 rows
print("\nFirst 5 rows:")
print(train_data.head())

# checking target variable distribution
print("\nTarget variable (price_range) distribution:")
print(train_data['price_range'].value_counts())

# ===========================================
# 2. Data Preprocessing (5 steps)
# ===========================================
print("\n" + "=" * 50)
print("2. DATA PREPROCESSING")
print("=" * 50)

# Step 1: checking missing values
print("\nStep 1: Checking missing values")
print("Missing values in training data:", train_data.isnull().sum().sum())
print("Missing values in test data:", test_data.isnull().sum().sum())

# Step 2: outlier detection using IQR method
print("\nStep 2: Outlier detection")

def find_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return len(outliers)

# check outliers in numeric columns
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('price_range')

print("Outliers found in each column:")
for col in numeric_cols:
    outlier_count = find_outliers(train_data, col)
    if outlier_count > 0:
        print(f"{col}: {outlier_count} outliers")

# Step 3: feature engineering - creating new features
print("\nStep 3: Feature engineering")
train_data['total_camera'] = train_data['pc'] + train_data['fc']
train_data['screen_size'] = train_data['sc_h'] * train_data['sc_w']
train_data['pixels'] = train_data['px_height'] * train_data['px_width']
train_data['battery_per_weight'] = train_data['battery_power'] / (train_data['mobile_wt'] + 1)

# same for test data
test_data['total_camera'] = test_data['pc'] + test_data['fc']
test_data['screen_size'] = test_data['sc_h'] * test_data['sc_w']
test_data['pixels'] = test_data['px_height'] * test_data['px_width']
test_data['battery_per_weight'] = test_data['battery_power'] / (test_data['mobile_wt'] + 1)

print("New features created: total_camera, screen_size, pixels, battery_per_weight")
print("New shape:", train_data.shape)

# Step 4: checking encoding of categorical/binary columns
print("\nStep 4: Checking encoding of categorical columns")
binary_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
print("Binary columns are already encoded as 0 and 1:")
for col in binary_cols:
    print(f"{col}: {train_data[col].unique()}")

# Step 5: splitting data for training and validation
print("\nStep 5: Splitting data for training and validation")
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# remove id from test if exists
if 'id' in test_data.columns:
    test_data = test_data.drop('id', axis=1)

# train test split - use stratified split for balanced classes
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Data split done! (Scaling will be done in pipeline)")

# ===========================================
# 3. Pipeline Creation
# ===========================================
print("\n" + "=" * 50)
print("3. PIPELINE CREATION")
print("=" * 50)

# creating pipeline that integrates preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Preprocessing step: feature scaling
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))  # Model
])

print("Pipeline created with preprocessing and model integration:")
print(pipeline)

# ===========================================
# 4. Primary Model Selection
# ===========================================
print("\n" + "=" * 50)
print("4. PRIMARY MODEL SELECTION")
print("=" * 50)

print("""
I am using Random Forest Classifier because:
- It works well for classification problems with multiple classes
- It can handle both numeric and categorical features
- It is not sensitive to outliers
- It gives good accuracy without much tuning
- It can show which features are important
""")

# ===========================================
# 5. Model Training
# ===========================================
print("\n" + "=" * 50)
print("5. MODEL TRAINING")
print("=" * 50)

# training the pipeline
pipeline.fit(X_train, y_train)

print("Model trained!")

# checking training accuracy
train_acc = pipeline.score(X_train, y_train)
print(f"Training accuracy: {train_acc:.4f}")

# feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': pipeline.named_steps['classifier'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 important features:")
print(feature_importance.head(10))

# ===========================================
# 6. Cross-Validation
# ===========================================
print("\n" + "=" * 50)
print("6. CROSS-VALIDATION")
print("=" * 50)

# cross validation on training data
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-validation scores for each fold:")
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.4f}")

print(f"\nMean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# ===========================================
# 7. Hyperparameter Tuning
# ===========================================
print("\n" + "=" * 50)
print("7. HYPERPARAMETER TUNING")
print("=" * 50)

# parameters to test (with regularization to prevent overfitting)
param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [8, 10],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4]
}

print("Parameters being tested (with regularization):")
for param, values in param_grid.items():
    print(f"{param}: {values}")

# grid search - only on training data to avoid data leakage
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

# ===========================================
# 8. Best Model Selection
# ===========================================
print("\n" + "=" * 50)
print("8. BEST MODEL SELECTION")
print("=" * 50)

# best model
best_model = grid_search.best_estimator_

print("Best model:")
print(best_model)

# get the classifier from pipeline
best_classifier = best_model.named_steps['classifier']
print(f"\nBest model settings:")
print(f"n_estimators: {best_classifier.n_estimators}")
print(f"max_depth: {best_classifier.max_depth}")
print(f"min_samples_split: {best_classifier.min_samples_split}")
print(f"min_samples_leaf: {best_classifier.min_samples_leaf}")

# ===========================================
# 9. Model Performance Evaluation
# ===========================================
print("\n" + "=" * 50)
print("9. MODEL PERFORMANCE EVALUATION")
print("=" * 50)

# check training vs validation accuracy to detect overfitting
train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# predictions on validation set
y_pred = best_model.predict(X_val)

# accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")


# precision, recall, f1
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(cm)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# ===========================================
# Save model as picked file
# ===========================================



with open('mobile_priced_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)