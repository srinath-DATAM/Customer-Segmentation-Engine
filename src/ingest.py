import pandas as pd 
import os 
def load_csv(file_path):
    df=pd.read_csv(file_path,encoding="latin")
    print("Dataset loaded successfully")
    return df
def clean_up_csv(df):
    print("handling missing values before cleanup")
    print(df.isnull().sum())
    # 1. Separate Numeric and Categorical columns for targeted cleaning
    # Numeric columns cleaning (e.g., handling negative quantities or prices)
    print("\nCleaning numeric columns...")
    # Removing rows where Quantity or UnitPrice is less than or equal to 0 (returns/errors)
    df=df[(df["Quantity"]>0)&(df["UnitPrice"]>0)]
    # 2. Categorical / ID columns cleaning
    print("Cleaning categorical & ID columns...")
    # Drop rows where Cu CustomerIDstomerID is missing (crucial for customer segmentation)
    df=df.dropna(subset=["CustomerID"])
    # Convert to integer format (Kaggle dataset usually has it as float due to NaNs)
    df["CustomerID"]=df["CustomerID"].astype(int)
    print("\nMissing values after cleanup:")
    print(df.isnull().sum())
    print(f"\nCleaned Dataset Total Rows: {df.shape[0]:,}")
    return df
if __name__ == "__main__":
    # Path to the raw dataset inside data/raw/
    filepath=os.path.join("data","raw","OnlineRetail.csv")
    # Step 1: Load CSV
    raw_file=load_csv(filepath)
    # Step 2: Clean CSV (Numeric & Categorical separation)
    cleaned_data=clean_up_csv(raw_file)
    #Save processed data to data/processed/ folder
    output_path = os.path.join("data", "processed", "cleaned_retail.csv")
    cleaned_data.to_csv(output_path, index=False)
    print(f"\nSaved cleaned file to: {output_path}")

