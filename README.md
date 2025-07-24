🧠 Feature Engineering:
------------------------
- No manual feature creation.
- Applied automated feature selection using LightGBM’s feature importances.
- Data preprocessing (e.g., handling NaNs, scaling) was assumed to be handled upstream in `processed_train.csv` and `processed_test.csv`.

📁 Files Included:
-------------------
1. `04_train_lightgbm.py` - Training script using top 50 features per target.
2. `05_predict_lightgbm.py` - Prediction script using trained models and feature subsets.
3. `top50_features_per_target.json` - JSON file mapping each target to its top 50 features.
4. `README.txt` - This explanation of the approach.

📌 Folder Structure Assumptions:
-------------------------------
- Train file: `data/processed_train.csv`
- Test file: `data/processed_test.csv`
- Models are saved in: `models/lightgbm_models/`
- Feature importance files are read from: `models/lightgbm_models/top_features/`

🎯 Status:
----------
- Models trained and validated.
- Submission CSV (`lightgbm_top50_submission.csv`) generated successfully.
- Current best leaderboard score: **67.64110**
- Next steps: Hyperparameter tuning using Optuna and ensembling.
