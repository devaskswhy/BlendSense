import pandas as pd
import lightgbm as lgb
import numpy as np
import os

# === Load test data ===
test = pd.read_csv("data/processed_test.csv")
test_ids = pd.read_csv("data/test.csv")["ID"]

# Drop ID column from test data if exists
if "ID" in test.columns:
    test = test.drop(columns=["ID"])

# === Initialize DataFrame for predictions ===
preds = pd.DataFrame()

# === Predict for each target ===
for i in range(1, 11):
    print(f"Predicting BlendProperty{i}...")

    # Load trained model
    model_path = f"models/lgb_target_{i}.txt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = lgb.Booster(model_file=model_path)

    # Load top features for this target
    feature_path = f"models/importance_target_{i}.csv"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature importance file not found: {feature_path}")
    top_features_df = pd.read_csv(feature_path)
    top_features = top_features_df["feature"].tolist()[:40]  # ⬅️ Limit to top 40 features only


    # Filter features to those that exist in test data
    top_features = [f for f in top_features if f in test.columns]
    X_test = test[top_features]

    # ✅ DEBUGGING: Check for feature mismatch
    missing = [f for f in top_features_df["feature"].tolist() if f not in test.columns]
    print(f"➡️  Target {i}: Using {len(top_features)} features")
    if missing:
        print(f"⚠️  Missing features in test data: {missing}")
    print(f"✅ Test shape for target {i}: {X_test.shape}")

    # Predict and store
    preds[f"BlendProperty{i}"] = model.predict(X_test)

# === Save predictions ===
submission_file = pd.concat([test_ids, preds], axis=1)
submission_file.to_csv("submission.csv", index=False, float_format="%.6f")

# Final output info
print("✅ Predictions saved to submission.csv")
print("📊 Columns:", submission_file.columns.tolist())
print("📐 Shape:", submission_file.shape)
