import os
import urllib.request
import ssl
import pandas as pd

def download_and_augment():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    os.makedirs(raw_dir, exist_ok=True)
    
    xls_path = os.path.join(raw_dir, "default_of_credit_card_clients.xls")
    csv_path = os.path.join(raw_dir, "augmented_credit_card_clients.csv")
    
    if not os.path.exists(xls_path):
        print("Downloading original UCI dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, xls_path)
        print("Original dataset downloaded successfully.")
        
    df = pd.read_excel(xls_path, header=1)
    
    if "ID" in df.columns:
        df = df.set_index("ID")
        
    print(f"Original shape: {df.shape}")
    
    augmented_df = df.sample(n=96000, replace=True, random_state=42)
    
    augmented_df = augmented_df.reset_index(drop=True)
    augmented_df.index.name = "ID"
    augmented_df = augmented_df.reset_index()
    
    augmented_df.to_csv(csv_path, index=False)
    print(f"Augmented dataset saved to {csv_path} with shape {augmented_df.shape}")

if __name__ == "__main__":
    download_and_augment()
