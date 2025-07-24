import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define paths
base_path = "data"
test_file = os.path.join(base_path, "test.csv")
processed_test_file = os.path.join(base_path, "processed_test.csv")
scaler_file = os.path.join(base_path, "scaler.pkl")

# Load test data and scaler
test_df = pd.read_csv(test_file)
scaler = joblib.load(scaler_file)

# Drop ID if it exists
if "ID" in test_df.columns:
    test_ids = test_df["ID"]  # save for final submission
    test_df = test_df.drop(columns=["ID"])
else:
    test_ids = None

# Scale test data using fitted scaler
X_test_scaled = scaler.transform(test_df)

# Convert to DataFrame
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=test_df.columns)

# If IDs were present, add them back (useful for submission files)
if test_ids is not None:
    X_test_scaled_df.insert(0, "ID", test_ids)

# Save
X_test_scaled_df.to_csv(processed_test_file, index=False)
print(f"Processed test data saved to: {processed_test_file}")
