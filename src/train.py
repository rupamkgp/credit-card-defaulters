import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_baselines(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss", n_jobs=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def tune_hyperparameters(model_name, X_train, y_train):
    if model_name == "XGBoost":
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
        param_dist = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "n_estimators": [100, 200, 300],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5]
        }
    elif model_name == "LightGBM":
        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
        param_dist = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9, -1],
            "n_estimators": [100, 200, 300],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5]
        }
    else:
        raise ValueError("Tuning only supported for XGBoost or LightGBM in this pipeline.")
        
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"Tuning {model_name}...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)
    return search.best_estimator_, search.best_params_
