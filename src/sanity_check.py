import pandas as pd 
import os 
file_path=os.path.join("data","processed","rfm_scaled.csv")
df=pd.read_csv(file_path)
# 2. Check for missing values (Must be 0 across all columns)
print(df.isnull().sum())
# 3. Check shape (Total customers after outlier removal)
print(f"\nTotal Clean Customers & Features: {df.shape}")
# 4. Quick statistical summary (Check if scaling worked - mean should be close to 0 for scaled columns)
print("\nData Summary Statistics:")
print(df[['Scaled_Recency', 'Scaled_Frequency', 'Scaled_Monetary']].describe())