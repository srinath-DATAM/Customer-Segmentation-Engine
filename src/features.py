import pandas as pd 
import os 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def rfm(input_path):
    df = pd.read_csv(input_path)
    print("Loading cleaned retail data...")
    
    # Convert InvoiceDate to datetime format
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # Create TotalPrice for monetary calculation
    df["Totalprice"] = df["Quantity"] * df["UnitPrice"]
    
    # Setting up reference date for Recency
    snap_shot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    # RFM calculation 
    print("Aggregating into RFM metrics...")
    rfm_df = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snap_shot_date - x.max()).days, # Recency
        "InvoiceNo": "nunique",                                   # Frequency
        "Totalprice": "sum"                                       # Monetary
    }).reset_index()
    
    rfm_df.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    print("RFM calculation completed successfully")
    
    # 5. Log Transformation (Correct column names matching)
    rfm_df["Log_Recency"] = np.log1p(rfm_df["Recency"])
    rfm_df["Log_Frequency"] = np.log1p(rfm_df["Frequency"])
    rfm_df["Log_Monetary"] = np.log1p(rfm_df["Monetary"])
    
    # 6. Standardization (StandardScaler)
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    scaled_cols = ['Log_Recency', 'Log_Frequency', 'Log_Monetary']
    
    rfm_df[['Scaled_Recency', 'Scaled_Frequency', 'Scaled_Monetary']] = scaler.fit_transform(rfm_df[scaled_cols])
    print(f"Total customers processed & scaled: {rfm_df.shape[0]:,}")
    
    # 7. Anomaly Pre-filtering using Isolation Forest
    print("Filtering extreme outliers using Isolation Forest...")
    iso = IsolationForest(contamination=0.02, random_state=42)
    scaled_data = rfm_df[["Scaled_Recency", "Scaled_Frequency", "Scaled_Monetary"]]
    
    # IsolationForest predicts: 1 for normal, -1 for outliers
    rfm_df["outliers"] = iso.fit_predict(scaled_data)
    
    # Keep only normal customers (outliers == 1)
    clean_rfm = rfm_df[rfm_df["outliers"] == 1].drop(columns=["outliers"]).reset_index(drop=True)
    
    print(f"Total customers before filtering: {rfm_df.shape[0]:,}")
    print(f"Total customers processed & cleaned: {clean_rfm.shape[0]:,}")

    return clean_rfm

if __name__ == "__main__":
    input_path = os.path.join("data", "processed", "cleaned_retail.csv")
    final_rfm = rfm(input_path)
    print(final_rfm.head())
    
    # Saving final clean scaled RFM output
    output_path = os.path.join("data", "processed", "rfm_scaled.csv")
    final_rfm.to_csv(output_path, index=False)
    print(f"Saved final scaled & filtered RFM file to: {output_path}")