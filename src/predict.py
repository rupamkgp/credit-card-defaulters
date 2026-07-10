import os
import joblib
import pandas as pd
import numpy as np

def load_prediction_artifacts(models_dir):
    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "features.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    
    return model, scaler, features

def predict_default(input_dict, model, scaler, feature_names):
    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Feature Engineering
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    
    df["Total_Bill"] = df[bill_cols].sum(axis=1)
    df["Total_Payment"] = df[pay_cols].sum(axis=1)
    df["Bill_Difference"] = df["Total_Bill"] - df["Total_Payment"]
    
    df["Average_Bill"] = df[bill_cols].mean(axis=1)
    df["Average_Payment"] = df[pay_cols].mean(axis=1)
    
    df["Payment_Ratio"] = df["Average_Payment"] / (df["Average_Bill"].replace(0, np.nan))
    df["Payment_Ratio"] = df["Payment_Ratio"].fillna(0)
    
    df["Bill_Utilization"] = df["Average_Bill"] / (df["LIMIT_BAL"].replace(0, np.nan))
    df["Bill_Utilization"] = df["Bill_Utilization"].fillna(0)
    
    df["Credit_Utilization"] = df["BILL_AMT1"] / (df["LIMIT_BAL"].replace(0, np.nan))
    df["Credit_Utilization"] = df["Credit_Utilization"].fillna(0)
    
    df["Payment_Delay_Count"] = df[delay_cols].apply(lambda row: sum(1 for x in row if x > 0), axis=1)
    df["Late_Payment_Frequency"] = df["Payment_Delay_Count"] / 6.0
    
    # Reorder columns to match training features exactly
    df = df[feature_names]
    
    # Scale continuous variables
    cols_to_scale = ["LIMIT_BAL", "AGE", "Total_Bill", "Total_Payment", "Bill_Difference", 
                     "Average_Bill", "Average_Payment", "Payment_Ratio", "Bill_Utilization", 
                     "Credit_Utilization", "Payment_Delay_Count", "Late_Payment_Frequency"]
    cols_to_scale += bill_cols
    cols_to_scale += pay_cols
    
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # Predict
    prob = model.predict_proba(df_scaled)[0, 1]
    pred = int(model.predict(df_scaled)[0])
    
    # Determine risk level
    if prob < 0.3:
        risk_level = "Low Risk"
    elif prob < 0.7:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
        
    # Generate recommendations
    recommendations = []
    if risk_level == "High Risk":
        recommendations.append("Immediate action required: Cut down credit card spending to prevent further balance accumulation.")
        recommendations.append("Prioritize clearing overdue statements to lower payment delay counts.")
        recommendations.append("Consider calling the credit issuer to discuss debt restructuring options.")
    elif risk_level == "Medium Risk":
        recommendations.append("Monitor utilization: Try to keep your credit utilization below 30%.")
        recommendations.append("Set automated payments to prevent missing payment deadlines.")
        recommendations.append("Pay more than the minimum balance whenever possible to reduce total bill accumulation.")
    else:
        recommendations.append("Keep up the excellent credit habits! Pay bill statements in full and maintain a low utilization rate.")
        recommendations.append("Periodically review your statements for any unauthorized transactions.")
        
    return {
        "probability": float(prob),
        "prediction": pred,
        "risk_level": risk_level,
        "recommendations": recommendations
    }
