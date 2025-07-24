Shell.ai Hackathon 2025 – Level 1 Submission
=============================================

🔍 Problem Statement:
---------------------
The task was to build a machine learning model to predict 10 blend properties (BlendProperty1 to BlendProperty10) using fuel component composition and physicochemical property features.

📌 Approach:
------------
1. **Baseline Modeling**:
   - Initial models were trained using `MultiOutputRegressor` with `Ridge` and then `LightGBM`.
   - Each property was modeled independently using a LightGBM Regressor.

2. **Feature Importance + Selection**:
   - Used `feature_importances_` from LightGBM to rank features for each target.
   - Selected top 50 features per target and stored them in `top50_features_per_target.json`.

3. **Model Training**:
   - Used 5-Fold Cross-Validation for robust performance estimation.
   - Trained 10 × 5 = 50 LightGBM models (one per target and fold) using the selected top features.

4. **Prediction Strategy**:
   - For each test sample, averaged predictions from the 5 models per target.
   - Final submission generated with predictions for all 10 blend properties.

⚙️ Tools & Libraries:
---------------------
- Python 3.10+
- NumPy, Pandas
- Scikit-learn
- LightGBM
- Optuna (for future tuning)
- joblib (for model serialization)

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

Prepared by: [Your Name]
Date: July 2025
