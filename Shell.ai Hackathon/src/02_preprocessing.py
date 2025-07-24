import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Set paths
data_dir = "data"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
processed_train_path = os.path.join(data_dir, "processed_train.csv")
scaler_path = os.path.join(data_dir, "scaler.pkl")

# Load the train data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Drop ID column
train_df = train_df.drop(columns=["ID"], errors='ignore')
test_ids = test_df["ID"]
test_df = test_df.drop(columns=["ID"])

# Separate targets
targets = [col for col in train_df.columns if col.startswith("BlendProperty")]
X_train = train_df.drop(columns=targets)
y_train = train_df[targets]

# Combine train and test for feature engineering
combined_df = pd.concat([X_train, test_df], axis=0)
print(f"Combined shape before feature engineering: {combined_df.shape}")

# ========== ✅ Step 1: Add statistical features across components ==========

# Extract numerical property names
property_names = [f"Property{i}" for i in range(1, 11)]

# For each property, calculate mean, std, min, max across all 5 components
for prop in property_names:
    cols = [f"Component{j}_{prop}" for j in range(1, 6)]
    combined_df[f"{prop}_mean"] = combined_df[cols].mean(axis=1)
    combined_df[f"{prop}_std"] = combined_df[cols].std(axis=1)
    combined_df[f"{prop}_min"] = combined_df[cols].min(axis=1)
    combined_df[f"{prop}_max"] = combined_df[cols].max(axis=1)

print(f"Combined shape after statistical feature engineering: {combined_df.shape}")

# ========== ✅ Step 2: Scaling ==========

# Scale features
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined_df)
combined_scaled_df = pd.DataFrame(combined_scaled, columns=combined_df.columns)

# Split back into train and test
X_train_processed = combined_scaled_df.iloc[:len(train_df)]
X_test_processed = combined_scaled_df.iloc[len(train_df):]

# Combine with targets
train_processed = pd.concat([X_train_processed, y_train.reset_index(drop=True)], axis=1)

# Save processed data
train_processed.to_csv(processed_train_path, index=False)
X_test_processed["ID"] = test_ids.values
X_test_processed.to_csv(os.path.join(data_dir, "processed_test.csv"), index=False)

# Save scaler
joblib.dump(scaler, scaler_path)

print("✅ Preprocessing complete with statistical features added.")
print(f"Processed train shape: {train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")