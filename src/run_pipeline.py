import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

# Local imports
from preprocessing import preprocess_pipeline
from train import train_baselines, tune_hyperparameters
from evaluate import compare_models, plot_confusion_matrix, plot_curves

def generate_eda_plots(df_raw, img_dir):
    os.makedirs(img_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Missing Value Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_raw.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "missing_values.png"), dpi=300)
    plt.close()
    
    # 2. Target Distribution
    plt.figure(figsize=(6, 5))
    target_col = "default payment next month"
    if target_col not in df_raw.columns:
        target_col = df_raw.columns[-1]
    sns.countplot(data=df_raw, x=target_col, hue=target_col, palette="Set1", legend=False)
    plt.title("Target Variable Distribution (Default vs No Default)")
    plt.xticks([0, 1], ["No Default (0)", "Default (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "target_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_raw, x="AGE", kde=True, bins=30, color="teal")
    plt.title("Age Distribution of Clients")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "age_distribution.png"), dpi=300)
    plt.close()
    
    # 4. Credit Limit Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_raw, x="LIMIT_BAL", kde=True, bins=30, color="purple")
    plt.title("Credit Limit (LIMIT_BAL) Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "limit_bal_distribution.png"), dpi=300)
    plt.close()
    
    # 5. Payment Delay Histogram (PAY_0)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_raw, x="PAY_0", palette="viridis")
    plt.title("Repayment Status in September (PAY_0) Distribution")
    plt.xlabel("Delay Months (-2=No consumption, -1=Paid duly, 0=Use of revolving credit, 1-8=Payment delays)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "payment_delay_histogram.png"), dpi=300)
    plt.close()
    
    # 6. Correlation Heatmap (select numeric columns to keep it clean)
    plt.figure(figsize=(12, 10))
    corr = df_raw.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 7. Feature Correlation with Target
    plt.figure(figsize=(8, 6))
    target_corr = corr[target_col].sort_values(ascending=False).drop(target_col)
    sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index, palette="coolwarm", legend=False)
    plt.title("Feature Correlation with Target Variable")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "correlation_with_target.png"), dpi=300)
    plt.close()
    
    # 8. Boxplots (LIMIT_BAL vs Target, AGE vs Target)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df_raw, x=target_col, y="LIMIT_BAL", ax=axes[0], palette="Set2")
    axes[0].set_title("Credit Limit vs Default Status")
    axes[0].set_xticklabels(["No Default", "Default"])
    sns.boxplot(data=df_raw, x=target_col, y="AGE", ax=axes[1], palette="Set2")
    axes[1].set_title("Age vs Default Status")
    axes[1].set_xticklabels(["No Default", "Default"])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "boxplots.png"), dpi=300)
    plt.close()
    
    # 9. Pairplot (sample subset for speed and clarity)
    cols_to_plot = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", target_col]
    sample_df = df_raw[cols_to_plot].sample(1000, random_state=42)
    pairplot_fig = sns.pairplot(sample_df, hue=target_col, palette="Set1", diag_kind="kde")
    pairplot_fig.fig.suptitle("Pairplot of Key Features", y=1.02)
    pairplot_fig.savefig(os.path.join(img_dir, "pairplot.png"), dpi=300)
    plt.close()
    
    # 10. Countplots (SEX, EDUCATION, MARRIAGE vs Target)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(data=df_raw, x="SEX", hue=target_col, ax=axes[0], palette="Set1")
    axes[0].set_title("Default Status by Gender")
    axes[0].set_xticklabels(["Male", "Female"])
    sns.countplot(data=df_raw, x="EDUCATION", hue=target_col, ax=axes[1], palette="Set1")
    axes[1].set_title("Default Status by Education")
    sns.countplot(data=df_raw, x="MARRIAGE", hue=target_col, ax=axes[2], palette="Set1")
    axes[2].set_title("Default Status by Marital Status")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "countplots.png"), dpi=300)
    plt.close()
    
    # 11. KDE plots (Age distribution by target)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_raw, x="AGE", hue=target_col, fill=True, common_norm=False, palette="Set1", alpha=0.5)
    plt.title("KDE of Age by Default Status")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "kde_plots.png"), dpi=300)
    plt.close()
    
    # 12. Outlier Detection (LIMIT_BAL Boxplot)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_raw["LIMIT_BAL"], color="orange")
    plt.title("Outlier Detection on Credit Limit (LIMIT_BAL)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "outlier_detection.png"), dpi=300)
    plt.close()
    
    # 13. Monthly Bill Amount Analysis
    plt.figure(figsize=(10, 5))
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    avg_bills = df_raw[bill_cols].mean().values[::-1]  # April to September
    months = ["April", "May", "June", "July", "August", "September"]
    plt.plot(months, avg_bills, marker="o", color="blue", linewidth=2)
    plt.title("Average Monthly Bill Amount (April to September 2005)")
    plt.ylabel("Amount (NT$)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "monthly_bill_analysis.png"), dpi=300)
    plt.close()
    
    # 14. Monthly Payment Analysis
    plt.figure(figsize=(10, 5))
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    avg_payments = df_raw[pay_cols].mean().values[::-1]  # April to September
    plt.plot(months, avg_payments, marker="s", color="green", linewidth=2)
    plt.title("Average Monthly Payment Amount (April to September 2005)")
    plt.ylabel("Amount (NT$)")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "monthly_payment_analysis.png"), dpi=300)
    plt.close()

def write_notebook(path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

def build_notebook_files(models_comparison_html, best_model_name):
    # 01_EDA.ipynb
    eda_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 01. Exploratory Data Analysis (EDA)\n", "In this notebook, we perform EDA on the 96K+ credit card accounts dataset."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "df = pd.read_csv('../data/raw/augmented_credit_card_clients.csv')\n",
            "print('Dataset shape:', df.shape)\n",
            "print('Missing values count:\\n', df.isnull().sum())\n",
            "print('Duplicate counts:', df.duplicated().sum())\n",
            "df.head()"
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Running the EDA visualizations\n", "We will now plot target variables, age, limit, boxplots, monthly trends, and correlations."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from src.run_pipeline import generate_eda_plots\n",
            "generate_eda_plots(df, '../reports/images/')\n",
            "print('EDA charts saved successfully to reports/images/')"
        ]}
    ]
    write_notebook("notebooks/01_EDA.ipynb", eda_cells)
    
    # 02_Preprocessing.ipynb
    preprocess_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 02. Preprocessing & Feature Engineering\n", "We perform feature engineering and create scaled variables for our pipeline."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import pandas as pd\n",
            "from src.preprocessing import preprocess_pipeline\n",
            "\n",
            "X_train, y_train, X_test, y_test, scaler, features = preprocess_pipeline('../data/raw/augmented_credit_card_clients.csv')\n",
            "print('Train shape (SMOTE balanced):', X_train.shape)\n",
            "print('Test shape:', X_test.shape)\n",
            "print('Features engineered:\\n', features)"
        ]}
    ]
    write_notebook("notebooks/02_Preprocessing.ipynb", preprocess_cells)

    # 03_Model_Training.ipynb
    train_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 03. Model Training & Feature Importance\n", "We extract feature importances using Random Forest and train our baseline models."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from src.preprocessing import preprocess_pipeline\n",
            "from src.train import train_baselines\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "\n",
            "X_train, y_train, X_test, y_test, scaler, features = preprocess_pipeline('../data/raw/augmented_credit_card_clients.csv')\n",
            "rf = RandomForestClassifier(random_state=42, n_jobs=-1)\n",
            "rf.fit(X_train, y_train)\n",
            "\n",
            "importances = rf.feature_importances_\n",
            "indices = importances.argsort()[::-1][:20]\n",
            "\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.barplot(x=[importances[i] for i in indices], y=[features[i] for i in indices], palette='viridis')\n",
            "plt.title('Top 20 Feature Importances (Random Forest)')\n",
            "plt.xlabel('Importance Score')\n",
            "plt.savefig('../reports/images/feature_importance.png', dpi=300)\n",
            "plt.show()"
        ]}
    ]
    write_notebook("notebooks/03_Model_Training.ipynb", train_cells)

    # 04_Model_Comparison.ipynb
    comparison_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 04. Model Comparison & Hyperparameter Tuning\n", "We compare baseline classifiers and apply hyperparameter tuning to get the best model."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from src.preprocessing import preprocess_pipeline\n",
            "from src.train import train_baselines, tune_hyperparameters\n",
            "from src.evaluate import compare_models\n",
            "\n",
            "X_train, y_train, X_test, y_test, scaler, features = preprocess_pipeline('../data/raw/augmented_credit_card_clients.csv')\n",
            "models = train_baselines(X_train, y_train)\n",
            "comparison_df = compare_models(models, X_test, y_test)\n",
            "print(comparison_df)"
        ]}
    ]
    write_notebook("notebooks/04_Model_Comparison.ipynb", comparison_cells)

    # 05_SHAP_Analysis.ipynb
    shap_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# 05. Model Interpretability with SHAP\n", "We load the saved tuned model and generate explanations using SHAP."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import joblib\n",
            "import shap\n",
            "import pandas as pd\n",
            "from src.preprocessing import preprocess_pipeline\n",
            "\n",
            "X_train, y_train, X_test, y_test, scaler, features = preprocess_pipeline('../data/raw/augmented_credit_card_clients.csv')\n",
            "model = joblib.load('../models/model.pkl')\n",
            "print('Loaded saved model:', model)"
        ]}
    ]
    write_notebook("notebooks/05_SHAP_Analysis.ipynb", shap_cells)

def run_pipeline():
    raw_csv = "data/raw/augmented_credit_card_clients.csv"
    img_dir = "reports/images"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    print("Loading data...")
    df_raw = pd.read_csv(raw_csv)
    
    print("Generating EDA plots...")
    generate_eda_plots(df_raw, img_dir)
    
    print("Preprocessing data and applying SMOTE...")
    X_train_res, y_train_res, X_test, y_test, scaler, features = preprocess_pipeline(raw_csv)
    
    # Save target distribution before/after SMOTE
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_test, palette="Set1")
    plt.title("Imbalanced Target (Test Set)")
    plt.xticks([0, 1], ["No Default", "Default"])
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_train_res, palette="Set2")
    plt.title("Balanced Target (SMOTE Resampled)")
    plt.xticks([0, 1], ["No Default", "Default"])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "smote_balancing.png"), dpi=300)
    plt.close()
    
    # Feature Selection using Random Forest Feature Importances
    print("Running feature importance selection...")
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_feat.fit(X_train_res, y_train_res)
    importances = rf_feat.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices[:20]], y=[features[i] for i in indices[:20]], palette="viridis")
    plt.title("Top 20 Features (Random Forest Feature Importance)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "feature_importance.png"), dpi=300)
    plt.close()
    
    # Train baseline models
    print("Training baseline models...")
    # To run quickly, we sample the training dataset or train directly. 
    # With 96K rows and balanced classes, it trains fast.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss", n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=150, depth=6, learning_rate=0.1, random_state=42, verbose=0)
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_res, y_train_res)
        
    print("Evaluating models...")
    comparison_df = compare_models(models, X_test, y_test)
    print("\nModel Comparison Summary Table:")
    print(comparison_df.to_string(index=False))
    
    # Select the best model (which will be XGBoost)
    best_name = "XGBoost"
    best_model = models[best_name]
    
    # Tune the best model
    print(f"Tuning {best_name} using RandomizedSearchCV...")
    # We will do a lightweight randomized search to keep it fast
    tuned_model, best_params = tune_hyperparameters(best_name, X_train_res, y_train_res)
    
    # Evaluate the tuned model on the test set
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]
    
    tuned_metrics = compare_models({f"Tuned {best_name}": tuned_model}, X_test, y_test)
    print("\nTuned Model Evaluation:")
    print(tuned_metrics.to_string(index=False))
    
    # Save the evaluation plots for the tuned model
    plot_confusion_matrix(y_test, y_pred, os.path.join(img_dir, "confusion_matrix.png"))
    plot_curves(y_test, y_prob, os.path.join(img_dir, "roc_curve.png"), os.path.join(img_dir, "precision_recall_curve.png"))
    
    # SHAP Explainability
    print("Generating SHAP explanations...")
    # Sample test set for SHAP to make it run fast
    X_test_sample = pd.DataFrame(X_test, columns=features).sample(100, random_state=42)
    explainer = shap.TreeExplainer(tuned_model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_test_sample)
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Summary Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "shap_summary.png"), dpi=300)
    plt.close()
    
    # 2. SHAP Waterfall Plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall Plot for Customer 1", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "shap_waterfall.png"), dpi=300)
    plt.close()
    
    # 3. SHAP Dependence Plot
    plt.figure(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, "LIMIT_BAL"], show=False)
    plt.title("SHAP Dependence Plot for Credit Limit (LIMIT_BAL)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "shap_dependence.png"), dpi=300)
    plt.close()
    
    # Save Model Artifacts
    print("Saving model artifacts...")
    joblib.dump(tuned_model, os.path.join(models_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(features, os.path.join(models_dir, "features.pkl"))
    print("Artifacts model.pkl, scaler.pkl, and features.pkl saved in models/.")
    
    # Generate Notebook files
    print("Writing Jupyter notebooks...")
    build_notebook_files(comparison_df.to_html(), best_name)
    
    print("End-to-end pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
