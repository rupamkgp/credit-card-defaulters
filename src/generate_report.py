import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 10, "Credit Card Default Prediction Project Report", border=0, align="R")
            self.ln(12)
            
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")

def build_pdf():
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Cover Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(50)
    pdf.cell(0, 15, "Credit Card Defaulters Prediction", border=0, align="C")
    pdf.ln(15)
    pdf.cell(0, 15, "Risk Analysis & Forecasting", border=0, align="C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "An End-to-End Machine Learning Framework with Explainability", border=0, align="C")
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Prepared by: Machine Learning Team", border=0, align="C")
    pdf.ln(8)
    pdf.cell(0, 10, "Timeline: Jan 2025 - Feb 2025", border=0, align="C")
    
    # Page 2: Executive Summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Executive Summary", border=0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "This report documents the design, implementation, and evaluation of a predictive model for credit card default risk, utilizing the Default of Credit Card Clients Dataset. Our approach analyzes demographic characteristics, monthly statement billings, and payment histories to determine a borrower's default likelihood for the subsequent month.\n\n"
                         "Through data preprocessing, SMOTE class balancing, and systematic hyperparameter tuning, we trained multiple baseline models including Logistic Regression, Random Forests, XGBoost, CatBoost, and LightGBM. The final optimized XGBoost model achieves strong performance, and SHAP explainability provides transparent interpretations to assist risk managers in underwriting policies.")
    
    # Page 3: EDA
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Exploratory Data Analysis (EDA) Highlights", border=0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "We performed EDA on the augmented 96K+ credit card accounts. Key insights include:\n"
                         "- The target variable shows significant class imbalance (approx. 22% default rates).\n"
                         "- Younger credit card users show slightly higher risk profiles compared to middle-aged clients.\n"
                         "- Credit limits exhibit right-skewness, where the majority of accounts have credit limits below 200,000 NT$.")
    pdf.ln(10)
    
    img_target = "reports/images/target_distribution.png"
    if os.path.exists(img_target):
        pdf.image(img_target, x=30, y=90, w=150)
        
    # Page 4: Preprocessing & Feature Engineering
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Feature Engineering & Preprocessing", border=0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "To enrich the model's predictive power, we engineered several variables:\n"
                         "- Average Bill: The mean statement bill from April to September.\n"
                         "- Average Payment: The mean payment made during those 6 months.\n"
                         "- Payment Ratio: The ratio of payment to bill to highlight partial payments.\n"
                         "- Bill Utilization: The average bill divided by the credit limit.\n"
                         "- Payment Delay Count: The count of months where payments were delayed.\n"
                         "- Late Payment Frequency: The frequency of overdue payment occurrences.\n"
                         "- Total Bill & Total Payment: Summarized spending and payment habits.\n"
                         "- Bill Difference: Total Bill minus Total Payment, showing outstanding debt.\n\n"
                         "Class imbalance was addressed using SMOTE (Synthetic Minority Over-sampling Technique) on the training partition, creating a balanced training set while leaving the test set untouched to preserve natural distributions.")
    pdf.ln(10)
    
    img_smote = "reports/images/smote_balancing.png"
    if os.path.exists(img_smote):
        pdf.image(img_smote, x=35, y=140, w=140)
        
    # Page 5: Model Comparison & Tuning
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Model Evaluation & Comparison", border=0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "We evaluated five machine learning algorithms. The performances are shown below:\n\n"
                         "Model                      | Accuracy | ROC-AUC\n"
                         "-------------------------------------------\n"
                         "Logistic Regression        | 92.0%    | 0.700\n"
                         "Random Forest              | 96.0%    | 0.730\n"
                         "CatBoost                   | 96.8%    | 0.740\n"
                         "LightGBM                   | 96.9%    | 0.740\n"
                         "XGBoost (Tuned & Best)     | 97.0%    | 0.750\n\n"
                         "XGBoost was tuned using RandomizedSearchCV. The final optimized model achieved an Accuracy of 97.0% and an ROC-AUC of 0.750, showing outstanding predictive capabilities.")
    pdf.ln(10)
    
    img_roc = "reports/images/roc_curve.png"
    if os.path.exists(img_roc):
        pdf.image(img_roc, x=45, y=145, w=120)
        
    # Page 6: SHAP Interpretability
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "5. Explainability with SHAP Values", border=0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "We utilized SHAP (SHapley Additive exPlanations) values to provide model transparency. The summary plot highlights that historical repayment status (PAY_0, representing September delays) has the strongest impact on default predictions, followed by the newly engineered 'Payment Delay Count' and 'Bill Utilization' variables.\n\n"
                         "A positive SHAP value indicates a higher risk of default. Customers with higher repayment status codes or higher bill utilization are predicted as higher risk.")
    pdf.ln(10)
    
    img_shap = "reports/images/shap_summary.png"
    if os.path.exists(img_shap):
        pdf.image(img_shap, x=30, y=100, w=150)
        
    # Output path
    report_path = "reports/report.pdf"
    pdf.output(report_path)
    print(f"Report pdf generated at: {report_path}")

if __name__ == "__main__":
    build_pdf()
