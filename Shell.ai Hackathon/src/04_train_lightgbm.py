import pandas as pd
import lightgbm as lgb
import os

# Load processed training data
df = pd.read_csv("data/processed_train.csv")

# Prepare directory for saving models
os.makedirs("models", exist_ok=True)

# Train separate LightGBM models using top N features for each target
TOP_N = 40

for i in range(1, 11):
    target_col = f"BlendProperty{i}"
    print(f"Training model for {target_col}...")

    # Load top features from importance file
    importance_path = f"models/importance_target_{i}.csv"
    importance_df = pd.read_csv(importance_path)
    top_features = importance_df["feature"].head(TOP_N).tolist()

    # Prepare train data
    X = df[top_features]
    y = df[target_col]

    # Create dataset and train
    train_data = lgb.Dataset(X, label=y)

    model = lgb.train(
        {"objective": "regression", "metric": "mae"},
        train_data,
        num_boost_round=1000
    )


    # Save model
    model.save_model(f"models/lgb_target_{i}.txt")

print("✅ All models trained and saved using top features.")
