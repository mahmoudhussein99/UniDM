import pandas as pd
import os
# Load the CSV file
DATASET=os.getenv("DATASET")
path_full=f"/scratch/mhussein/UniDM/dataset/datasets/entity_matching/structured/{DATASET}"
if not os.path.exists(os.path.join(path_full,"test_original.csv")):
    # if doesn't exist then make a copy firstly
    df = pd.read_csv(os.path.join(path_full,"test.csv"),index_col=False)  
    df.to_csv(os.path.join(path_full,"test_original.csv"), index=False)
df = pd.read_csv(os.path.join(path_full,"test_original.csv"),index_col=False)  # Replace with your file name

# Randomly sample 110 rows
df_sampled = df.sample(frac=0.1, random_state=42)  # Set seed for reproducibility

# Save the downsampled dataset
df_sampled.to_csv(os.path.join(path_full,"test.csv"), index=False)
print("Downsampling complete! Saved as 'test.csv'.")
