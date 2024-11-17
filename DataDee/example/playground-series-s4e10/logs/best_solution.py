import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
import lightgbm as lgb

# Load data
train_data = pd.read_csv("../../../../playground-series-s4e10/train.csv")
test_data = pd.read_csv("../../../../playground-series-s4e10/test.csv")

# Separate features and target
X = train_data.drop(["loan_status", "id"], axis=1)
y = train_data["loan_status"]
X_test = test_data.drop("id", axis=1)

# Define categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# Interaction terms - adding polynomial features for top numerical features based on domain knowledge
top_numerical_features = numerical_features[
    :2
]  # Assume the first two are the top features for simplicity
poly_transformer = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False
)

# Update preprocessor to include polynomial features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("scaler", StandardScaler()), ("poly", poly_transformer)]),
            top_numerical_features,
        ),
        (
            "num_rest",
            StandardScaler(),
            list(set(numerical_features) - set(top_numerical_features)),
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Define the model
model = lgb.LGBMClassifier()

# Create and fit the pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

# Parameters grid
param_grid = {
    "classifier__num_leaves": [31, 50, 70],
    "classifier__max_depth": [10, 20, 30],
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring=make_scorer(roc_auc_score, needs_proba=True)
)

# Fit grid search
grid_search.fit(X, y)

# Best model
best_pipeline = grid_search.best_estimator_

# Predict on validation set
y_pred = best_pipeline.predict_proba(X)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y, y_pred)
print(f"Best GridSearch AUC: {auc_score}")

# Predict on test set
test_preds = best_pipeline.predict_proba(X_test)[:, 1]

# Save predictions to submission.csv
submission = pd.DataFrame({"id": test_data["id"], "loan_status": test_preds})
submission.to_csv("./submission.csv", index=False)
