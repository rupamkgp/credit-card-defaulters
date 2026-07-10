import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    df = pd.read_csv(filepath)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df

def clean_data(df):
    df = df.drop_duplicates()
    
    # Clean EDUCATION: group 0, 5, 6 with 4 (others/unknown)
    df.loc[df["EDUCATION"].isin([0, 5, 6]), "EDUCATION"] = 4
    
    # Clean MARRIAGE: group 0 with 3 (others)
    df.loc[df["MARRIAGE"] == 0, "MARRIAGE"] = 3
    
    # Fill any missing values with median/mode if present
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def engineer_features(df):
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    
    df["Total_Bill"] = df[bill_cols].sum(axis=1)
    df["Total_Payment"] = df[pay_cols].sum(axis=1)
    df["Bill_Difference"] = df["Total_Bill"] - df["Total_Payment"]
    
    df["Average_Bill"] = df[bill_cols].mean(axis=1)
    df["Average_Payment"] = df[pay_cols].mean(axis=1)
    
    # Handle division by zero
    df["Payment_Ratio"] = df["Average_Payment"] / (df["Average_Bill"].replace(0, np.nan))
    df["Payment_Ratio"] = df["Payment_Ratio"].fillna(0)
    
    df["Bill_Utilization"] = df["Average_Bill"] / (df["LIMIT_BAL"].replace(0, np.nan))
    df["Bill_Utilization"] = df["Bill_Utilization"].fillna(0)
    
    df["Credit_Utilization"] = df["BILL_AMT1"] / (df["LIMIT_BAL"].replace(0, np.nan))
    df["Credit_Utilization"] = df["Credit_Utilization"].fillna(0)
    
    # Delay features
    df["Payment_Delay_Count"] = df[delay_cols].apply(lambda row: sum(1 for x in row if x > 0), axis=1)
    df["Late_Payment_Frequency"] = df["Payment_Delay_Count"] / 6.0
    
    return df

def preprocess_pipeline(filepath, target_col="default payment next month"):
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale continuous variables
    cols_to_scale = ["LIMIT_BAL", "AGE", "Total_Bill", "Total_Payment", "Bill_Difference", 
                     "Average_Bill", "Average_Payment", "Payment_Ratio", "Bill_Utilization", 
                     "Credit_Utilization", "Payment_Delay_Count", "Late_Payment_Frequency"]
    
    # Add bill and payment amounts to scale
    cols_to_scale += [f"BILL_AMT{i}" for i in range(1, 7)]
    cols_to_scale += [f"PAY_AMT{i}" for i in range(1, 7)]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Apply SMOTE to handle imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_res, y_train_res, X_test_scaled, y_test, scaler, X.columns.tolist()
